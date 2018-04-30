from .calc_node import CalcNode
import numpy as np


class Mul(CalcNode):
    def __init__(self, node1: CalcNode, node2: CalcNode):
        self._node1 = node1 # x
        self._node2 = node2 # W
        # 要検証
        # print(self._node1.forward().shape)
        # print(self._node2.forward().shape)
        self.output = np.dot(self._node1.forward(), self._node2.forward())
        # print(self._output.shape)
        # print('*'*30)
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
            self._node1.backward(np.dot(self.grad, self._node2.output.T))
            self._node2.backward(np.outer(self._node1.output, self.grad))
            # print(np.dot(self.grad, self._node2.output.T).shape)
            # print(np.outer(self._node1.output, self.grad).shape)
            # print('*'*30)