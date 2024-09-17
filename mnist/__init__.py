from typing import TYPE_CHECKING

from mnist import idx
from mnist.perceptron import *

__all__ = [
    "idx",
    "Perceptron"
]

if TYPE_CHECKING:
    from mnist.typing import *

    __all__ += [
        "Float", "FloatArray", "FloatLike", "TFloatLike"
    ]
