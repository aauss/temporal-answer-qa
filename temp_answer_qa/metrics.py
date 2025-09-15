import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(
        absolute_error=lambda df: df["error_numeric"].abs(),
        exact_match=lambda df: df["error_numeric"] == 0,
        symmetric_absolute_percentage_error=lambda df: symmetric_absolute_percentage_error(
            df["response_digit"], df["label_digit"]
        ),
        absolute_deviance=lambda df: calculate_absolute_deviance(df),
        absolute_scaled_error=lambda df: absolute_scaled_error(
            df["response_digit"], df["label_digit"], df["absolute_deviance"]
        ),
    )
    return df


def calculate_absolute_deviance(df: pd.DataFrame) -> pd.Series:
    # Cluster then calculate MASE
    df.loc[:, "mean_answer_and_label"] = df.groupby("answer_temporal_unit")[
        "label_digit"
    ].transform(cluster_centroids_for_1d_arr)
    df[["mean_answer_per_cluster", "cluster_label"]] = pd.DataFrame(
        df["mean_answer_and_label"].tolist(), index=df.index
    )
    df.drop(columns="mean_answer_and_label", inplace=True)
    return (df["label_digit"] - df["mean_answer_per_cluster"]).abs()


def cluster_centroids_for_1d_arr(series):
    arr = series.values
    min_cluster_size = int(len(arr) * 0.3)
    if min_cluster_size == 1:
        return zip(np.array([np.mean(arr)] * len(arr)), np.array([0] * len(arr)))
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size, allow_single_cluster=True, store_centers="centroid"
    )
    hdb.fit(arr.reshape(-1, 1))
    # Assign noisy labels to the closest centroid
    mapping = np.argmin(np.abs(hdb.centroids_ - arr[hdb.labels_ == -1]), axis=0)
    mask = hdb.labels_ == -1
    hdb.labels_[mask] = mapping
    # Replace label by respective centroid for each data point
    return zip(hdb.centroids_.reshape(-1)[hdb.labels_], hdb.labels_)


def symmetric_absolute_percentage_error(
    predictions: np.ndarray, references: np.ndarray
) -> np.ndarray:
    """Calculate the symmetric absolute percentage error.

    Missing values must be np.nan.
    Args:
        predictions: Model predictions
        references: Reference values

    Returns:
        Symmetric absolute percentage error between predictions and references.
    """
    absolute_errors = abs(predictions - references)

    zero_mask = (abs(predictions) == 0) & (abs(references) == 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sape = np.where(
            zero_mask, 0, (absolute_errors / (abs(predictions) + abs(references))) * 100
        )
    return np.nan_to_num(sape, nan=100)


def absolute_scaled_error(
    predictions: np.ndarray, references: np.ndarray, deviances: np.ndarray
) -> np.ndarray:
    """Calculate the absolute scaled error

    Deviance must be larger than 0.
    Missing values must be np.nan.
    Args:
        predictions: Model predictions
        references: Reference values
        deviances: Deviances by which to scale the absolute errors

    Returns:
        Absolute error scaled by deviances.
    """
    absolute_errors = abs(predictions - references)
    zero_mask = (absolute_errors == 0) & (deviances == 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ase = np.where(zero_mask, 0, (absolute_errors / abs(deviances)))
    return ase
