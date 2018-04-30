import numpy as np
from calc_node import *
from optimizer import *


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, labels):
        self.input_size = input_size
        self.output_size = len(labels)
        self.hidden_sizes = hidden_sizes
        self.layer_sizes = (input_size, *hidden_sizes, len(labels))
        # print(self.layer_sizes)
        self.labels = labels
        self.label_index = {labels[i]: i for i in range(len(labels))}
        self.variables = self._init_variables()
        self.optimizer = SGD(self.variables)

    def _init_variables(self):
        variables = {}
        for i in range(len(self.layer_sizes)-1):
            variables['w{}'.format(i)] = Variable(np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]))
            variables['b{}'.format(i)] = Variable(np.random.randn(self.layer_sizes[i + 1]))
        return variables

    def _calculate(self, x: np.ndarray):
        graph = Variable(x)
        for i in range(len(self.layer_sizes)-1):
            # Affine
            graph = Add(
                Mul(
                    graph,
                    self.variables['w{}'.format(i)]
                ),
                self.variables['b{}'.format(i)]
            )
            if i < len(self.layer_sizes)-2:
                graph = Sigmoid(graph)
        return graph

    def fit(self, x: np.ndarray, label, update=True):
        graph = self._calculate(x)

        t_node = self.t_node(label)
        graph = SoftmaxLoss(graph, t_node)
        loss = graph.forward()

        graph.backward(1)
        if update:
            self.optimizer.update()
        return loss

    def predict(self, x):
        graph = self._calculate(x)
        y = graph.forward()
        max_index = np.argmax(y)
        return self.labels[max_index]

    def t_node(self, label):
        index = self.label_index[label]
        t = np.zeros(self.output_size)
        t[index] = 1.0
        return Variable(t)

    def numeric_gradient(self, x, label):
        eps = 1e-6
        target = self.variables['w1'].output
        target[0][0] += eps
        print(target[0][0])
        loss_plus = self.fit(x, label, False)
        # print(loss_plus)

        target[0][0] -= 2 * eps
        loss_minus = self.fit(x, label, False)
        print(target[0][0])
        # print(loss_minus)

        target[0][0] += eps

        return (loss_plus - loss_minus) / (2 * eps)

    def gradient_check(self, x, label):
        loss = self.fit(x, label, False)
        backprop_grad = self.variables['w1'].grad[0][0]
        numeric_grad = self.numeric_gradient(x, label)

        print('backprop: {} numeric: {}'.format(backprop_grad, numeric_grad))

        

