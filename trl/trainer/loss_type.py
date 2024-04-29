from enum import Enum

class LossType(Enum):
    NONE = 0
    XENTROPY = 1
    REBEL = 2
    DPO = 3