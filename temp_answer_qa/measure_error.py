from datetime import timedelta, datetime

import numpy as np
import pandas as pd


def try_calc_error(predicted, expected):
    try:
        return predicted - expected
    except (TypeError, ValueError):
        # Some expected answers are dates, lists, or predicted answers single years
        return None


def tot_measure_error(responses: pd.DataFrame) -> pd.DataFrame:
    tot_error_measurer = ToTErrorMeasurer()
    return responses.assign(
        error=lambda df: df.apply(
            lambda row: try_calc_error(row["response_numeric"], row["label_numeric"]), axis=1
        ),
        error_numeric=lambda df: df.apply(
            lambda row: tot_error_measurer.error_to_digit(
                row["error"], row["answer_temporal_unit"]
            ),
            axis=1,
        ),
        response_digit=lambda df: df.apply(
            lambda row: tot_error_measurer.model_response_to_digit(
                row["response_numeric"], row["answer_temporal_unit"]
            ),
            axis=1,
        ),
        label_digit=lambda df: df.apply(
            lambda row: tot_error_measurer.model_response_to_digit(
                row["label_numeric"], row["answer_temporal_unit"]
            ),
            axis=1,
        ),
    )


class ToTErrorMeasurer:
    def error_to_digit(self, error, answer_temporal_unit: str):
        if pd.isna(error) or pd.isna(answer_temporal_unit):
            return np.nan
        if answer_temporal_unit == "seconds" and isinstance(error, timedelta):
            return error.total_seconds()
        elif answer_temporal_unit == "minutes" and isinstance(error, timedelta):
            return error.total_seconds() / 60
        elif answer_temporal_unit in ("days", "months", "years"):
            return error
        elif (answer_temporal_unit == "date") and isinstance(error, pd.Timedelta):
            return error.days
        else:
            return np.nan

    def model_response_to_digit(self, response_numeric, answer_temporal_unit: str):
        if pd.isna(answer_temporal_unit):
            return np.nan
        if answer_temporal_unit == "seconds" and isinstance(response_numeric, timedelta):
            return response_numeric.total_seconds()
        elif answer_temporal_unit == "minutes" and isinstance(response_numeric, timedelta):
            return response_numeric.total_seconds() / 60
        elif answer_temporal_unit == "date" and isinstance(response_numeric, pd.Timestamp):
            return response_numeric.timestamp()
        elif (answer_temporal_unit in (["days", "months", "years"])) and isinstance(
            response_numeric, (int, float)
        ):
            return response_numeric
        else:
            return np.nan


def ttqa_measure_error(responses: pd.DataFrame) -> pd.DataFrame:
    ttqa_error_measurer = TTQAMeasurer()
    return responses.assign(
        error=lambda df: df.apply(
            lambda row: try_calc_error(row["response_numeric"], row["label_numeric"]), axis=1
        ),
        error_numeric=lambda df: df.apply(
            lambda row: ttqa_error_measurer.error_to_digit(
                row["error"], row["answer_temporal_unit"]
            ),
            axis=1,
        ),
        response_digit=lambda df: df.apply(
            lambda row: ttqa_error_measurer.model_response_to_digit(
                row["response_numeric"], row["answer_temporal_unit"]
            ),
            axis=1,
        ),
        label_digit=lambda df: df.apply(
            lambda row: ttqa_error_measurer.model_response_to_digit(
                row["label_numeric"], row["answer_temporal_unit"]
            ),
            axis=1,
        ),
    )


class TTQAMeasurer:
    def error_to_digit(self, error, answer_temporal_unit: str):
        if pd.isna(error) or pd.isna(answer_temporal_unit):
            return np.nan
        if answer_temporal_unit == "date" and isinstance(error, (pd.Timedelta, timedelta)):
            return error.days
        elif answer_temporal_unit in ["years", "date_years", "days", "months"] and isinstance(
            error, (int, float)
        ):
            return error
        else:
            return np.nan

    def model_response_to_digit(self, response_numeric, answer_temporal_unit: str):
        if pd.isna(answer_temporal_unit):
            return np.nan
        if answer_temporal_unit == "date" and isinstance(
            response_numeric, (pd.Timestamp, datetime)
        ):
            return response_numeric.timestamp()
        elif answer_temporal_unit in ["years", "date_years", "days", "months"] and isinstance(
            response_numeric, (int, float)
        ):
            return response_numeric
        else:
            return np.nan
