from .calc_node import CalcNode
import numpy as np


class SoftmaxLoss(CalcNode):
    def __init__(self, node: CalcNode, node_t: CalcNode):
        self._node = node
        self._node_t = node_t
        t = node_t.forward()
        x = node.forward()
        exp_vector = np.exp(x)
        self._p = exp_vector / np.sum(exp_vector)
        self.output = - np.dot(x - np.log(sum(exp_vector)), t)
        self._count = 0
        self.grad = np.zeros_like(self.output)

    def forward(self):
        self._count += 1
        return self.output

    def backward(self, d):
        self._count -= 1
        self.grad += d
        if self._count == 0:
            # print('backprop : Softmax')
            self._node.backward(self.grad * (self._p - self._node_t.output))
