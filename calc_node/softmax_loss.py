from .calc_node import CalcNode
import numpy as np


class SoftmaxLoss(CalcNode):
    def __init__(self, node: CalcNode, node_t: CalcNode):
        self._node = node
        self._node_t = node_t
        exp_vector = np.exp(node.forward())
        self._p = exp_vector / np.sum(exp_vector)
        self._output = - np.dot(node.forward - np.log(sum(exp_vector)), node_t.forward())
        self._count = 0
        self.grad = np.zeros_like(self._output)

    def forward(self):
        self._count += 1
        return self._output

    def backward(self, d):
        self._count -= 1
        self.grad += d
        if self._count == 0:
            self._node.backward(self.grad * (self._p - self._node_t.forward()))