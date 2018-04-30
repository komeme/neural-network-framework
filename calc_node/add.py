from .calc_node import CalcNode
import numpy as np


class Add(CalcNode):
    def __init__(self, node1: CalcNode, node2: CalcNode):
        self._node1 = node1
        self._node2 = node2
        self.output = node1.forward() + node2.forward()
        self._count = 0
        self.grad = np.zeros_like(self.output)

    def forward(self):
        self._count += 1
        return self.output

    def backward(self, d):
        if d.shape != self.output.shape:
            raise Exception()
        self._count -= 1
        self.grad += d
        if self._count == 0:
            self._node1.backward(self.grad)
            self._node2.backward(self.grad)
