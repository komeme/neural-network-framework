import numpy as np
from collections import namedtuple
from network import *
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


class NetworkEvaluator:
    def __init__(self, network, train_data, test_data):
        self._nn = network
        self._train_data = train_data
        self._test_data = test_data
        self._accuracy_list = []
        self._loss_list = []

    def train(self, num_epoch, plot=False, save_path=None):
        for i in tqdm(range(num_epoch)):
            # Resultテーブル
            FitResult = namedtuple('FitResult', ('label', 'loss'))
            fit_results = []

            random_int = np.random.randint(0, len(self._train_data), size=10)
            for n in random_int:
                x, label = self._train_data[n]
                loss = self._nn.fit(np.array(x), label)
                # self._nn.gradient_check(x, label)
                fit_results.append(FitResult(label, loss))

            # 平均ロスを追加
            self._loss_list.append(np.average([result.loss for result in fit_results]))

        if save_path is not None:
            with open(save_path) as f:
                pickle.dump(self._nn, f)

        if plot:
            self._plot_loss()

    def train_and_test(self, num_epoch, plot=False, save_path=None):
        for i in tqdm(range(num_epoch)):
            # Resultテーブル
            Result = namedtuple('Result', ('actual', 'predicted'))
            results = []

            FitResult = namedtuple('FitResult', ('label', 'loss'))
            fit_results = []

            random_int = np.random.randint(0, len(self._train_data), size=100)
            for n in random_int:
                x, label = self._train_data[n]
                loss = self._nn.fit(np.array(x), label)
                fit_results.append(FitResult(label, loss))

            random_int = np.random.randint(0, len(self._test_data), size=100)
            for m in random_int:
                x, label = self._test_data[m]
                predicted = self._nn.predict(np.array(x))
                results.append(Result(label, predicted))

            # 平均ロスを追加
            self._loss_list.append(np.average([result.loss for result in fit_results]))

            # 正答率を追加
            correct = [result.actual == result.predicted for result in results].count(True)
            accuracy = 100.0 * correct / len(results)
            # self._print_console(i + 1, num_epoch, accuracy)
            self._accuracy_list.append(accuracy)

        if save_path is not None:
            with open(save_path, 'bw') as f:
                pickle.dump(self._nn, f)

        if plot:
            self._plot_accuracy()

    @staticmethod
    def _print_console(epoch, num_epoch, accuracy):
        print('epoch{}/{}: accuracy: {}[%]'.format(epoch, num_epoch, accuracy))

    def _plot_accuracy(self):
        x = range(len(self._accuracy_list))
        plt.plot(x, self._accuracy_list)
        plt.show()

    def _plot_loss(self):
        x = range(len(self._loss_list))
        plt.plot(x, self._loss_list)
        plt.show()



