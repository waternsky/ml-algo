import numpy as np
from typing import Callable
from code_for_hw02 import (
    test_perceptron,
    test_averaged_perceptron,
    test_linear_classifier,
    super_simple_separable as sss,
    super_simple_separable_through_origin as sssto
)


def perceptron(data: np.ndarray,
               labels: np.ndarray,
               params: dict[str, int] = {},
               hook: Callable | None = None) -> (np.ndarray, float):
    '''
    Parameters:
        data: np.ndarray
            d * n array, n data points of dimension d
        labels: np.ndarray
            1 * n array, row vector of labels corresponding to n datapoints
        params: dict[str, int]
            default = {}
            Parameters that algorithm might need like, number of iteration
        hook: Callable | None
            default = None
            For testing purpose, hook in some function to visualise

    Returns:
        hyperplane (th, th0): (np.ndarray, float)
            hyperplane after training on data set provided
    '''
    num_itr = params.get("num_itr", 100)
    th = np.full((data.shape[0], 1), 0.)
    th0 = 0.
    for i in range(num_itr):
        for j in range(data.shape[1]):
            vec = data[:, j:(j+1)]
            if labels[0][j] * (th.T @ vec + th0) <= 0:
                th = th + labels[0][j] * vec
                th0 = th0 + labels[0][j]
    return th, th0


def averaged_perceptron(data: np.ndarray,
                        labels: np.ndarray,
                        params: dict[str, int] = {},
                        hook: Callable | None = None) -> (np.ndarray, float):
    '''
    Perceptron can be sensitive to most recent data points. Averaged perceptron
    produces more stable output.
    Parameters:
        data: np.ndarray
            d * n array, n data points of dimension d
        labels: np.ndarray
            1 * n array, row vector of labels corresponding to n datapoints
        params: dict[str, int]
            default = {}
            Parameters that algorithm might need like, number of iteration
        hook: Callable | None
            default = None
            For testing purpose, hook in some function to visualise

    Returns:
        hyperplane (th, th0): (np.ndarray, float)
            hyperplane after training on data set provided
    '''
    num_itr = params.get("num_itr", 100)
    num_data = data.shape[1]
    th = np.full((data.shape[0], 1), 0.)
    avg_th = np.full((data.shape[0], 1), 0.)
    th0 = 0.
    avg_th0 = 0.
    for i in range(num_itr):
        for j in range(num_data):
            vec = data[:, j:(j+1)]
            if labels[0][j] * (th.T @ vec + th0) <= 0:
                th = th + labels[0][j] * vec
                th0 = th0 + labels[0][j]
            avg_th = avg_th + th
            avg_th0 = avg_th0 + th0
    return avg_th / (num_data * num_itr), avg_th0 / (num_data * num_itr)


def main() -> None:
    test_perceptron(perceptron)
    test_averaged_perceptron(averaged_perceptron)
    for datafn in (sss, sssto):
        data, labels = datafn()
        test_linear_classifier(datafn, perceptron, draw=True)


if __name__ == "__main__":
    main()
