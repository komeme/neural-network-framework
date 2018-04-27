from .calc_node import CalcNode
import numpy as np


class Mul(CalcNode):
    def __init__(self, node1: CalcNode, node2: CalcNode):
        self._node1 = node1
        self._node2 = node2
        # 要検証
        self._output = np.dot(self._node1.forward(), self._node2.forward())
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
            self._node1.backward(self._node2.forward() * self.grad)
            self._node2.backward(self._node1.forward() * self.grad)