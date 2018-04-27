from .calc_node import CalcNode
import numpy as np


class Variable(CalcNode):
    def __init__(self, var: np.ndarray):
        self.value = var
        self.grad = np.zeros_like(var)
        self._count = 0

    def forward(self):
        self._count += 1
        return self.value

    def backward(self, d):
        if d.shape != self.value.shape:
            raise Exception()
        self._count -= 1
        self.grad += d
