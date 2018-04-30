from .calc_node import CalcNode
import numpy as np


class Variable(CalcNode):
    def __init__(self, var: np.ndarray):
        self.output = var
        self.grad = np.zeros_like(var)
        self._count = 0

    def forward(self):
        self._count += 1
        return self.output

    def backward(self, d):
        if d.shape != self.output.shape:
            raise Exception()
        self._count -= 1
        self.grad += d
