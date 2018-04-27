from abc import ABC, abstractmethod
import numpy as np


class CalcNode(metaclass=ABC):
    @abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, d):
        raise NotImplementedError()
