import numpy as np
from numpy_basics import length, rv, cv


def signed_distance(p: np.ndarray, th: np.ndarray, th0: float) -> np.ndarray:
    '''
    Parameters:
        p: np.ndarray
            column vector (d * 1)  - a point p,in d dimension.
        th: np.ndarray
            column vector (d * 1) a hyperplane parameter, normal to plane.
        th0: float
            scalar value a hyperplane parameter.
        th.T @ x + th0 = 0

    Returns:
        distance: np.ndarray
            1 * 1 array of signed distance of point cv from
            plane th.T @ x + th0 = 0.
    '''
    return (th.T @ p + th0) / length(th)


def side_of_plane(p: np.ndarray, th: np.ndarray, th0: float) -> [-1, 0, 1]:
    return np.sign(th.T @ p + th0)


def score(data: np.ndarray, labels: np.ndarray,
          th: np.ndarray, th0: float) -> int:
    '''
    Parameters:
        data: np.ndarray
            d * n array of floats, n data points in d dimension.
        labels: np.ndarray
            1 * n array of elements in {-1, 0, 1}, representing target labels.
        th: np.ndarray
            column vector (d * 1) a hyperplane parameter, normal to plane.
        th0: float
            scalar value a hyperplane parameter.
        th.T @ x + th0 = 0

    Returns:
        score: int
            number of point for which labels is equal to output.
    '''
    out_labels = np.sign(th.T @ data + th0)
    return np.sum(out_labels == labels)


def best_separator(data: np.ndarray, labels: np.ndarray,
                   ths: np.ndarray, th0s: np.ndarray) -> (np.ndarray, float):
    '''
    Parameters:
        data: np.ndarray
            d * n array of floats, n data points in d dimension.
        labels: np.ndarray
            1 * n array of elements in {-1, 0, 1}, representing target labels.
        ths: np.ndarray
            d * m array of hyperplane parameter, m normal vetcors
        th0s: np.ndarray
            1 * m array of corresponding scalar for th

    Returns:
        best_separator: tuple(np.ndarray, float)
            best th & th0 among m choices available
    '''
    def score_mat(data, labels, ths, th0s):
        # out_labels_mat is m * n, th.T @ data is m * n,
        # each column of it is added by th0s.T which is m * 1
        # that is broadcasting opearator in numpy
        # out_labels_mat == labels, checks row wise,
        # to get m * n against 1 * n
        # after np.sum returning m * 1 (axis = 1) column vector
        # make sure to keepdims=True, to keep array (m, 1) not (m,)
        out_labels_mat = np.sign(ths.T @ data + th0s.T)
        return np.sum(out_labels_mat == labels, axis=1, keepdims=True)

    print(score_mat(data, labels, ths, th0s).shape)
    best_sep_idx = np.argmax(score_mat(data, labels, ths, th0s))
    return ths[:, best_sep_idx], th0s[:, best_sep_idx][0]


def main() -> None:
    x = np.array([[2], [3]])
    th = np.array([[3], [4]])
    th0 = 5
    ans = signed_distance(x, th, th0)
    assert ans == 4.6, "signed_distance is incorret"
    assert side_of_plane(x, th, th0) == 1, "side_of_plane is incorrect"

    data = np.transpose(np.array([[1, 2], [1, 3], [2, 1], [1, -1], [2, -1]]))
    labels = rv([-1, -1, 1, 1, 1])
    assert score(data, labels, cv([1, 1]), -2) == 1, "score is incorrect"
    assert score(data, labels, cv([-1, -1]), 2) == 4, "score is incorrect"
    assert score(data, labels, cv([0, 1]), 1) == 1, "score is incorrect"

    ths = np.transpose(np.array([[1, 1], [-1, -1], [0, 1]]))
    th0s = np.array([[-2, 2, 1]])
    best_th, best_th0 = best_separator(data, labels, ths, th0s)
    print(best_th, best_th0)
    assert (best_th == cv([-1, -1])).all(), "best_separator is incorrect"
    assert best_th0 == 2, "best_separator is incorrect"


if __name__ == "__main__":
    main()
