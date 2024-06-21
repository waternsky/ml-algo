import numpy as np
import math


def print_vec(fun: callable) -> None:
    def wrapper(*args, **kwargs) -> np.ndarray:
        print("Arguments provided: ", *args, **kwargs)
        print(f"Function name: {fun.__name__}")
        ans = fun(*args, **kwargs)
        print("Returns: ", ans)
        print("Shape: ", ans.shape)
        print("---------------------------------")
        return ans
    return wrapper


@print_vec
def rv(num_list: list[int]) -> np.ndarray:
    return np.array([num_list])


@print_vec
def cv(num_list: list[int]) -> np.ndarray:
    return np.reshape(num_list, (len(num_list), 1))


def length(column_vector: np.ndarray) -> int:
    return math.sqrt((column_vector.T @ column_vector)[0, 0])


@print_vec
def normalize(column_vector: np.ndarray) -> np.ndarray:
    return column_vector / length(column_vector)


def final_column(mat_2d: np.ndarray) -> np.ndarray:
    return mat_2d[:, -1].reshape((mat_2d.shape[0], 1))


def main() -> None:
    li = [1, 2, 3, 4]
    col_vec = cv(li)
    row_vec = rv(li)
    unit_vec = normalize(col_vec)
    print(length(col_vec))
    print(row_vec, unit_vec)
    mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    final_col = final_column(mat)
    print(final_col)


if __name__ == "__main__":
    main()
