import numpy as np
from hyperplanes import score
from typing import Callable
from perceptron import perceptron, averaged_perceptron
from code_for_hw02 import (
    test_eval_classifier,
    test_eval_learning_alg,
    test_xval_learning_alg,
    gen_flipped_lin_separable
)


def eval_classifier(learner: Callable,
                    data_train: np.ndarray,
                    labels_train: np.ndarray,
                    data_test: np.ndarray,
                    labels_test: np.ndarray) -> float:
    '''
    Evaluating particular classfier obtained from learning algo.

    Parameters:
        learner: Callable
            a function, such as perceptron or averaged perceptron
        data_train: np.ndarray
            training data
        labels_train: np.ndarray
            training labels
        data_test: np.ndarray
            test data
        labels_test: np.ndarray
            test lables

    Returns:
        percentage: float
            percenatage correct on new testing set as a float between 0. and 1.
    '''
    th, th0 = learner(data_train, labels_train)
    return score(data_test, labels_test, th, th0) / data_test.shape[1]


def eval_learning_alg(learner: Callable,
                      data_gen: Callable[[int], tuple[np.ndarray, np.ndarray]],
                      n_train: int,
                      n_test: int,
                      itr: int) -> float:
    '''
    Given a data source and learner checks accuracy of learning algo.

    Parameters:
        learner: Callable
            a function, such as perceptron or averaged perceptron
        data_gen: Callable[[int], tuple[np.ndarray, np.ndarray]]
            a data generator, given data size, returns (data, label)
        n_train: int
            size of training set
        n_test: int
            size of test sets
        itr: int
            number of iteration to average over

    Returns:
        accuracy: float
            average classification accuracy between 0. and 1.
    '''
    total: float = 0.
    for i in range(itr):
        data_train, labels_train = data_gen(n_train)
        data_test, labels_test = data_gen(n_test)
        total += eval_classifier(learner, data_train, labels_train,
                                 data_test, labels_test)
    return total / itr


def xval_learning_alg(learner: Callable,
                      data: np.ndarray,
                      labels: np.ndarray,
                      k: int) -> float:
    '''
    Cross validation, data of fixed size to evaluate learning algo.

    Parameters:
        learner: Callable
            a function, such as perceptron or averaged perceptron
        data: np.ndarray
            data set, d * n array
        labels: np.ndarray
            labels, 1 * n row vector
        k: int
            equal parts data should be divided into for cross validation

    Returns:
        accuracy: float
            average classification accuracy between 0. and 1.
    '''
    data_array = np.array_split(data, k, axis=1)
    label_array = np.array_split(labels, k, axis=1)
    total = 0.

    def join(arr: list[np.ndarray], i: int) -> np.ndarray:
        return np.concatenate(arr[0:i] + arr[(i+1):], axis=1)

    for i in range(k):
        data_train = join(data_array, i)
        label_train = join(label_array, i)
        total += eval_classifier(learner, data_train, label_train,
                                 data_array[i], label_array[i])
    return total / k


def main() -> None:
    test_eval_classifier(eval_classifier, perceptron)
    test_eval_learning_alg(eval_learning_alg, perceptron)
    test_xval_learning_alg(xval_learning_alg, perceptron)

    data_gen = gen_flipped_lin_separable(pflip=0.1)
    pans = eval_learning_alg(perceptron, data_gen, 20, 20, 100)
    avgpans = eval_learning_alg(averaged_perceptron, data_gen, 80, 20, 30)
    print(pans, avgpans)


if __name__ == "__main__":
    main()
