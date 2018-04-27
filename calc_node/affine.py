from .calc_node import CalcNode
from calc_node import *
import numpy as np


class Affine(CalcNode):
    def __init__(self, x, W, b):
        self._node = Add(Mul(W, x), b)
        self._count = 0
        self._output = self._node.forward()
        self.grad = np.zeros_like(self._output)

    def forward(self):
        self._count += 1
        return self._output

    def backward(self, d):
        self._node.backward(d)
