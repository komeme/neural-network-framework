from evaluator import NetworkEvaluator
from network import RNN
import gensim.models.keyedvectors as word2vec

def main():
    with open('data/wsj00-18.pos') as f:
        training_data = f.readlines()

    with open('data/wsj19-21.pos') as f:
        test_data = f.readlines()

    model = word2vec.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)

    print(model.wv.most_similar(positive=['woman', 'king'], negative=['man']))
    # print(model['hgoehgoehgoehg'])
    # print(len(model['hogehgoehgoe']))

    labels = ('NNP', ',', 'CD', 'NNS', 'JJ', 'MD', 'VB', 'DT', 'NN', 'IN', '.', 'VBZ', 'VBG', 'CC', 'VBD', 'VBN', 'RB', 'TO', 'PRP', 'RBR', 'WDT', 'VBP', 'RP', 'PRP$', 'JJS', 'POS', '``', 'EX', "''", 'WP', ':', 'JJR', 'WRB', '$', 'NNPS', 'WP$', '-LRB-', '-RRB-', 'PDT', 'RBS', 'FW', 'UH', 'SYM', 'LS', '#')

    rnn = RNN(300, 1000, labels)

    training_vector_data = [
        line for line in training_data
    ]

    test_vector_data = [
        line for line in test_data
    ]

    manager = NetworkEvaluator(rnn, training_vector_data, test_vector_data)


if __name__ == '__main__':
    main()