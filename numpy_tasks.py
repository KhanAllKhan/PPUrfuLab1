"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    if len(matrices) == 0:
        return np.zeros((0, 1))
    return np.sum(matrices @ vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return np.where(matrix > threshold, 1, 0)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(col).tolist() for col in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed

    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))

    row_mean = np.mean(matrix, axis=1)
    col_mean = np.mean(matrix, axis=0)
    row_var = np.var(matrix, axis=1)
    col_var = np.var(matrix, axis=0)

    return MatrixStatistics(
        matrix=matrix,
        row_mean=row_mean,
        column_mean=col_mean,
        row_dispersion=row_var,
        column_dispersion=col_var
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    # Cоздание шахматной сетки через сумму индексов осей
    grid = np.indices((rows, columns)).sum(axis=0)
    return np.where(grid % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)

    center_y, center_x = image_height // 2, image_width // 2

    # Границы прямоугольника с учетом четности размеров
    y_start = max(0, center_y - height // 2)
    y_end = min(image_height, center_y + (height + 1) // 2)
    x_start = max(0, center_x - width // 2)
    x_end = min(image_width, center_x + (width + 1) // 2)

    image[y_start:y_end, x_start:x_end] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)

    center_y, center_x = image_height // 2, image_width // 2

    y, x = np.ogrid[:image_height, :image_width]

    mask = ((x - center_x) ** 2) / (semi_axis_x ** 2) + ((y - center_y) ** 2) / (semi_axis_y ** 2) <= 1
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window

    mean = np.mean(values)
    var = np.var(values)
    std = np.std(values)

    left = values[:-2]
    mid = values[1:-1]
    right = values[2:]

    max_indices = np.where((mid > left) & (mid > right))[0] + 1
    min_indices = np.where((mid < left) & (mid < right))[0] + 1

    rolling_mean = np.convolve(values, np.ones(window) / window, mode='valid')

    return TimeSeriesStatistics(
        mean=mean,
        variance=var,
        standard_deviation=std,
        local_maxima_indices=max_indices,
        local_minima_indices=min_indices,
        rolling_mean=rolling_mean
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count

    if class_count is None:
        class_count = int(np.max(labels)) + 1

    encoded = np.zeros((len(labels), class_count), dtype=np.ndarray)

    encoded[np.arange(len(labels)), labels] = 1
    return encoded