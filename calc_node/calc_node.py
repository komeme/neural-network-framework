from abc import ABCMeta, abstractmethod


class CalcNode(metaclass=ABCMeta):
    @abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, d):
        raise NotImplementedError()
