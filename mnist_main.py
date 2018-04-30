from dataset.mnist import load_mnist
from evaluator import *
from network import NeuralNetwork


def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

    train_data = [(x, t) for x, t in zip(x_train, t_train)]
    test_data = [(x, t) for x, t in zip(x_test, t_test)]

    labels = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    nn = NeuralNetwork(784, (100,), labels)

    manager = NetworkEvaluator(nn, train_data, test_data)

    manager.train(num_epoch=10, plot=True)
    # manager.train_and_test(num_epoch=1, plot=True, save_path='data/nn2.pickle')

if __name__ == '__main__':
    main()