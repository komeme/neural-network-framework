import numpy as np
from optimizer import SGD
from calc_node import *

class RNN:
    def __init__(self, input_size, hidden_size, labels):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = len(labels)
        self.labels = labels
        self.label_index = {labels[i]: i for i in range(len(labels))}
        self.variables = self._init_variables()
        self.optimizer = SGD(self.variables)

    def _init_variables(self):
        variables = {
            'w': Variable(np.random.randn(self.input_size, self.hidden_size)),
            'w_rec': Variable(np.random.randn(self.hidden_size, self.hidden_size)),
            'b': Variable(np.random.randn(self.hidden_size)),
            'c': Variable(np.random.randn(self.output_size))
        }
        return variables

    def _calculate(self, input_list):
        hidden = Variable(np.zeros(self.hidden_size))
        output_list = []
        for x in input_list:
            hidden = Add(
                Mul(
                    Sigmoid(
                        Add(
                           Mul(
                               Variable(x),
                               self.variables['w']
                           ),
                           self.variables['b']
                        )
                    ),
                    hidden
                ),
                self.variables['c']
            )
            output_list.append(hidden)
        return output_list

    def fit(self, input_list, label, update=True):
        output_list = self._calculate(input_list)

        t_node = self.t_node(label)
        graph = np.zeros(self.hidden_size)
        for hidden in output_list:
            graph = Add(
                graph,
                SoftmaxLoss(hidden, t_node)
            )
        loss = graph.forward()
        graph.backward(1)
        if update:
            self.optimizer.update()
        return loss

    def predict(self, input_list):
        output_list = self._calculate(input_list)
        results = []

        for hidden in output_list:
            y = hidden.forward()
            max_index = np.argmax(y)
            results.append(self.labels[max_index])
        return results

    def t_node(self, label):
        index = self.label_index[label]
        t = np.zeros(self.output_size)
        t[index] = 1.0
        return Variable(t)