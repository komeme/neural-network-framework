import numpy as np


class SGD:
    def __init__(self, variables: dict, lr=0.1):
        self.variables = variables
        self.lr = lr

    def update(self):
        for key, var in self.variables.items():
            var.output -= self.lr * var.grad
            var.grad = np.zeros_like(var.grad)
