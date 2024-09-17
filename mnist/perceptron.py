from abc import ABC, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Self, overload
    
    from mnist.typing import *

__all__ = [
    "Perceptron"
]


class T:
    def __new__(cls, arr: np.ndarray) -> np.ndarray:
        return arr.swapaxes(-1, -2)


class Perceptron(ABC):

    def __init__(self, layer_b: "list[np.ndarray]", layer_w: "list[np.ndarray]"):
        assert len(layer_b) == len(layer_w)
        for b, w0, w1 in zip(layer_b, layer_w, layer_w[1:]):
            assert b.shape[1] == w0.shape[1] == w1.shape[0]
            assert b.shape[0] == 1

        self.layer_b = layer_b
        self.layer_w = layer_w
        self.n = len(self.layer_b)

    def save(self, filename: 'str') -> None:
        npz = {"n": np.int_(self.n)}
        for i in range(self.n):
            npz[f"b{i}"] = self.layer_b[i]
            npz[f"w{i}"] = self.layer_w[i]
        np.savez(filename, **npz)

    @classmethod
    def load(cls, filename: 'str') -> 'Self':
        bw = np.load(filename)
        b, w = [], []
        for i in range(bw["n"]):
            b += [bw[f"b{i}"]]
            w += [bw[f"w{i}"]]
        res = cls.__new__(cls)
        Perceptron.__init__(res, b, w)
        return res

    @abstractmethod
    def activate(self, x: "TFloatLike") -> "TFloatLike":
        pass

    @abstractmethod
    def activate_derivative(self, x: "TFloatLike") -> "TFloatLike":
        pass

    @abstractmethod
    def cost(self, y: "FloatArray", y0: "FloatArray") -> "Float":
        pass

    @abstractmethod
    def cost_derivative(self, y: "FloatArray", y0: "FloatArray") -> "FloatArray":
        pass

    if TYPE_CHECKING:
        @overload
        def __call__(self, x0: "FloatArray") -> "FloatArray": ...

        @overload
        def __call__(
                self,
                x0: "FloatArray",
                y0: "FloatArray"
        ) -> "tuple[list[FloatArray], list[FloatArray], float]": ...

    def __call__(self, x0, y0=None):
        assert x0.shape[-1] == self.layer_w[0].shape[0]
        if y0 is None:
            y = x0
            for b, w in zip(self.layer_b, self.layer_w):
                y = self.activate(b+y@w)
            return y
        assert y0.shape[-1] == self.layer_w[-1].shape[1]
        x = [x0]
        y = [x0]
        for b, w in zip(self.layer_b, self.layer_w):
            x += [b+y[-1]@w]
            y += [self.activate(x[-1])]
        n = self.n
        db_r = [self.cost_derivative(y[n], y0) * self.activate_derivative(x[n])]
        dw_r = [T(y[n-1])@db_r[-1]]
        for i in range(n-1, 0, -1):
            db_r += [db_r[-1]@T(self.layer_w[i])*self.activate_derivative(x[i])]
            dw_r += [T(y[i-1])@db_r[-1]]

        def avg(dl):
            for j in range(len(dl)):
                if dl[j].ndim > 2:
                    dl[j] = np.average(dl[j]
                                       .reshape((reduce(lambda x1, x2: x1 * x2, dl[j].shape[:-2]),
                                                 *dl[j].shape[-2:])), 0)
        avg(db_r)
        avg(dw_r)
        return reversed(db_r), reversed(dw_r), self.cost(y[-1], y0)
