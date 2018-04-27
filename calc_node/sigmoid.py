from .calc_node import CalcNode
import numpy as np


class Sigmoid(CalcNode):
    def __init__(self, node: CalcNode):
        self._node = node
        self._output = self._calc(node.forward())
        self._count = 0
        self.grad = np.zeros_like(self._output)

    def forward(self):
        self._count += 1
        return self._output

    def backward(self, d):
        if d.shape != self._output.shape:
            raise Exception()
        self._count -= 1
        self.grad += d
        if self._count == 0:
            self._node.backward(self.grad * self._output * (1. - self._output))

    @staticmethod
    def _calc(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))