from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeVar

    import numpy as np

    __all__ = [
        "Float", "FloatArray", "FloatLike", "TFloatLike"
    ]

    Float = float | np.floating
    FloatArray = np.ndarray[..., float | np.floating]
    FloatLike = Float | FloatArray
    TFloatLike = TypeVar("TFloatLike", bound=FloatLike)
