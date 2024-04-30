# REBEL Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from huggingface_hub.utils._deprecation import _deprecate_arguments
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from ..import_utils import is_peft_available, is_wandb_available
from ..models import PreTrainedModelWrapper, create_reference_model
from .rebel_config import REBELConfig
from .utils import (
    REBELDataCollatorWithPadding,
    RunningMoments,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


class REBELTrainer(Trainer):
    r"""
    Initialize REBELTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        args (`REBELConfig`):
            The REBEL config arguments to use for training.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "rebel"]

    @_deprecate_arguments(
        version="1.0.0",
        deprecated_args=[
            "pad_token_id",
            "disable_dropout",
            "dataset_num_proc",
            "model_init_kwargs",
            "ref_model_init_kwargs",
            "model_adapter_name",
            "ref_adapter_name",
            "force_use_ref_model",
            "eta",
        ],
        custom_message="Deprecated positional argument(s) used in REBELTrainer, please use the REBELConfig to set these arguments instead.",
    )
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[REBELConfig] = None,
        pad_token_id: Optional[int] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        peft_config: Optional[Dict] = None,
        disable_dropout: bool = True,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        force_use_ref_model: bool = False,
        eta: Optional[float] = None,
    ):
        if model_init_kwargs is not None:
            warnings.warn(
                "You passed `model_init_kwargs` to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.model_init_kwargs = model_init_kwargs

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_init_kwargs to the REBELTrainer/REBELConfig, but your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs
            model_init_kwargs["torch_dtype"] = (
                model_init_kwargs["torch_dtype"]
                if model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, model_init_kwargs["torch_dtype"])
            )

        if ref_model_init_kwargs is not None:
            warnings.warn(
                "You passed `ref_model_kwargs` to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.ref_model_init_kwargs = ref_model_init_kwargs

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_init_kwargs to the REBELTrainer/REBELConfig, but your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            ref_model_init_kwargs["torch_dtype"] = (
                ref_model_init_kwargs["torch_dtype"]
                if ref_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, ref_model_init_kwargs["torch_dtype"])
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the REBELTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the REBELTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if force_use_ref_model:
            warnings.warn(
                "You passed `force_use_ref_model` to the REBELTrainer, the value you passed will override the one in the `REBELConfig`."
            )
            args.force_use_ref_model = force_use_ref_model

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not args.force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with REBEL there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in REBELTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        if model_adapter_name is not None:
            warnings.warn(
                "You passed `model_adapter_name` to the REBELTrainer, the value you passed will override the one in the `REBELConfig`."
            )
            args.model_adapter_name = model_adapter_name
        self.model_adapter_name = args.model_adapter_name

        if ref_adapter_name is not None:
            warnings.warn(
                "You passed `ref_adapter_name` to the REBELTrainer, the value you passed will override the one in the `REBELConfig`."
            )
            args.ref_adapter_name = ref_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a REBEL dataset.")

        if pad_token_id is not None:
            warnings.warn(
                "You passed `pad_token_id` to the REBELTrainer, the value you passed will override the one in the `REBELConfig`."
            )
            args.pad_token_id = pad_token_id
        if data_collator is None:
            data_collator = REBELDataCollatorWithPadding(
                pad_token_id=args.pad_token_id,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using REBELDataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_rebel_data_collator = True
        else:
            self.use_rebel_data_collator = False

        if not disable_dropout:
            warnings.warn(
                "You passed `disable_dropout` to the REBELTrainer, the value you passed will override the one in the `REBELConfig`."
            )
            args.disable_dropout = disable_dropout
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)
                
        self.pad_token_id = args.pad_token_id
        self.tokenizer = tokenizer

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        if dataset_num_proc is not None:
            warnings.warn(
                "You passed `dataset_num_proc` to the REBELTrainer, the value you passed will override the one in the `REBELConfig`."
            )
            args.dataset_num_proc = dataset_num_proc
        self.dataset_num_proc = args.dataset_num_proc
        
        if eta is not None:
            warnings.warn(
                "You passed `eta` to the REBELTrainer, the value you passed will override the one in the `REBELConfig`."
            )
            self.eta = eta
        else:
            self.eta = args.eta

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset
            train_dataset = train_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        if self.ref_model is None:
            if not self.is_peft_model:
                raise ValueError(
                    "No reference model and model is not a Peft model."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def build_tokenized_answer(self, prompt: str, answer: str, rewards: Union[float, Dict[int, int]], player_ids: Optional[List[int]] = None):
        prompt_tokens = self.tokenizer(prompt)
        answer_tokens = self.tokenizer(answer)
        prompt_loss_codes = [LossType.NONE for i in range(len(prompt_tokens["input_ids"]))]
        
        if player_ids is None:
            assert isinstance(rewards, float)
            player_ids = [0 for i in range(len(answer_tokens["input_ids"]))]
            rewards = {0: rewards}
        else:
            assert isinstance(rewards, dict)
            assert all([player_id in rewards for player_id in player_ids])
        assert len(player_ids) == len(answer_tokens["input_ids"])
        
        player_id_to_loss_code = lambda player_id: LossType.XENTROPY if player_id == -1 else LossType.REBEL
        
        full_input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
        full_attention_mask = prompt_tokens["attention_mask"] + answer_tokens["attention_mask"]
        full_player_ids = [-1 for i in range(len(prompt_tokens["input_ids"]))] + player_ids
        full_loss_codes = prompt_loss_codes + [player_id_to_loss_code(player_id) for player_id in player_ids]
        input_ids = full_input_ids[:-1]
        attention_mask = full_attention_mask[:-1]
        target_ids = full_input_ids[1:]
        player_ids = full_player_ids[1:]
        loss_codes = full_loss_codes[1:]
        
        rewards = [rewards.get(i, 0) for i in range(max(rewards.keys()))] + [0]
        
        return dict(
            input_ids=np.array(input_ids),
            target_ids=np.array(target_ids),
            player_ids=np.array(player_ids),
            loss_codes=np.array(loss_codes),
            rewards=np.array(rewards),
            attention_mask=np.array(attention_mask),
        )

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row from a REBEL specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]
        chosen_rewards = feature.get("chosen_rewards", 1.)
        rejected_rewards = feature.get("rejected_rewards", 1.)
        chosen_player_ids = feature["chosen_player_ids"]
        rejected_player_ids = feature["rejected_player_ids"]
        
        for k, v in self.build_tokenized_answer(prompt, chosen, chosen_rewards, chosen_player_ids).items():
            batch[f"chosen_{k}"] = v
        for k, v in self.build_tokenized_answer(prompt, rejected, rejected_rewards, rejected_player_ids).items():
            batch[f"rejected_{k}"] = v

        return batch

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
            self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[torch.LongTensor]],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Values are tensors of shape (batch_size, ?).
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated batch tensors.
        """
        concatenated_batch = {}

        for k in sorted(batch): # so chosen_ keys always come first
            max_length = max(batch[k.replace("rejected", "chosen")].shape[1], batch[k.replace("chosen", "rejected")].shape[1])
            if k.endswith("player_ids"):
                pad_value = -1
            elif k.endswith("ids"):
                pad_value = self.pad_token_id
            elif k.endswith("mask"):
                pad_value = 0
            elif k.endswith("rewards"):
                pad_value = 0
            elif k.endswith("loss_codes"):
                pad_value = LossType.NONE
            else:
                raise ValueError(f"Unexpected key encountered in data collator: {k}")
            concatenated_key = k.replace("chosen", "concatenated").replace("rejected", "concatenated")
            padded = pad_to_length(batch[k], max_length, pad_value=pad_value)
            if concatenated_key not in concatenated_batch:
                concatenated_batch[concatenated_key] = padded
            else:
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        padded,
                    ),
                    dim=0,
                )

        return concatenated_batch

    def rebel_loss(
        self,
        policy_logits: torch.FloatTensor,
        reference_logits: torch.FloatTensor,
        concatenated_batch: Dict[str, Union[torch.LongTensor]],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the REBEL loss for a batch of policy and reference model log probabilities.

        Args:
            policy_logits: pre-softmax policy model outputs
            reference_logits: pre-softmax reference model outputs
            concatenated_batch: the concatenated batch tensors

        Returns:
            A tuple of four tensors: (xentropy, rebel, chosen_rewards, rejected_rewards).
            The xentropy tensor contains the cross entropy loss.
            The rebel tensor contains the rebel loss.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        assert policy_logits.shape[0] % 2 == 0
        
        split = lambda tensor: (tensor[:tensor.shape[0] // 2], tensor[tensor.shape[0] // 2:])
        policy_chosen_logits, policy_rejected_logits = split(policy_logits)
        reference_chosen_logits, reference_rejected_logits = split(reference_logits)
        chosen_target_ids, rejected_target_ids = split(concatenated_batch["concatenated_target_ids"])
        chosen_loss_codes, rejected_loss_codes = split(concatenated_batch["concatenated_loss_codes"])
        chosen_player_ids, rejected_player_ids = split(concatenated_batch["concatenated_player_ids"])
        chosen_rewards, rejected_rewards = split(concatenated_batch["concatenated_rewards"])
        
        compute_xentropy = lambda logits, targets, loss_codes: (F.cross_entropy(logits, targets, reduction='none') * (loss_codes == LossType.XENTROPY)).mean()
        xentropy = 0.5 * (
            compute_xentropy(policy_chosen_logits, chosen_target_ids, chosen_loss_codes) + \
            compute_xentropy(policy_rejected_logits, rejected_target_ids, rejected_loss_codes)
        )
        
        rebel = torch.zeros(1, dtype=policy_chosen_logits.dtype, device=policy_chosen_logits.device)
        policy_chosen_logits = torch.gather(F.log_softmax(policy_chosen_logits), dim=2, index=chosen_target_ids[..., None])
        policy_rejected_logits = torch.gather(F.log_softmax(policy_rejected_logits), dim=2, index=rejected_target_ids[..., None])
        reference_chosen_logits = torch.gather(F.log_softmax(reference_chosen_logits), dim=2, index=chosen_target_ids[..., None])
        reference_rejected_logits = torch.gather(F.log_softmax(reference_rejected_logits), dim=2, index=rejected_target_ids[..., None])
        chosen_diff = policy_chosen_logits - reference_chosen_logits
        rejected_diff = policy_rejected_logits - reference_rejected_logits
        
        impl_chosen_rewards = []
        impl_rejected_rewards = []
        player_ids = [player_id for player_id in list(chosen_player_ids.unique()) if player_id > -1] # chosen_player_ids.unique() is guaranteed to equal rejected_player_ids.unique()
        for player_id in player_ids:
            chosen_mask = chosen_player_ids == player_id
            rejected_mask = rejected_player_ids == player_id
            impl_chosen_reward = 1. / self.eta * (chosen_diff * (chosen_player_ids == player_id)).sum(-1)
            impl_rejected_reward = 1. / self.eta * (rejected_diff * (rejected_player_ids == player_id)).sum(-1)
            impl_regret = impl_chosen_reward - impl_rejected_reward
            actual_regret = chosen_rewards[:, player_id] - rejected_rewards[:, player_id]
            rebel = rebel + (actual_regret - impl_regret).pow(2).mean()
            impl_chosen_rewards.append(impl_chosen_reward.detach())
            impl_rejected_rewards.append(impl_rejected_reward.detach())
        rebel = rebel / len(player_ids)
        
        impl_chosen_rewards = torch.cat(impl_chosen_rewards, dim=0)
        impl_rejected_rewards = torch.cat(impl_rejected_rewards, dim=0)
        return xentropy, rebel, impl_chosen_rewards, impl_rejected_rewards

    def concatenated_forward(
        self, model: nn.Module, concatenated_batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> torch.FloatTensor:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
        ).logits
        
        return all_logits

    def get_batch_loss_metrics_(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """Compute the REBEL loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        
        concatenated_batch = self.concatenated_inputs(
            batch,
            device=self.accelerator.device,
        )

        policy_logits = self.concatenated_forward(model, concatenated_batch)


        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    reference_logits = self.concatenated_forward(self.model, concatenated_batch)
            else:
                reference_logits = self.concatenated_forward(self.ref_model, concatenated_batch)

        xentropy, rebel, chosen_rewards, rejected_rewards = self.rebel_loss(
            policy_logits, reference_logits, concatenated_batch,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}rebel/loss"] = rebel.detach().cpu()
        metrics[f"{prefix}xentropy/loss"] = xentropy.detach().cpu()

        return rebel + xentropy, metrics
    
    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        if train_eval == "train":
            return self.get_batch_loss_metrics_(model, batch)
        else:
            with torch.no_grad():
                return self.get_batch_loss_metrics_(model, batch)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_rebel_data_collator:
            warnings.warn(
                "compute_loss is only implemented for REBELDataCollatorWithPadding, and you passed a datacollator that is different than "
                "REBELDataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_rebel_data_collator:
            warnings.warn(
                "prediction_step is only implemented for REBELDataCollatorWithPadding, and you passed a datacollator that is different than "
                "REBELDataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "rebel" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
