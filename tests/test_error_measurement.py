from datetime import timedelta

import numpy as np
import pandas as pd

from temp_answer_qa.measure_error import tot_measure_error, ttqa_measure_error


def test_tot_error_measurement():
    test_df_tot = pd.DataFrame(
        {
            "response_numeric": [
                1,
                2,
                timedelta(seconds=3),
                pd.Timedelta("18:48:28"),
                None,
                pd.Timestamp("2022-01-02"),
                4,
                pd.Timedelta("18:38:22"),
                timedelta(seconds=120),
                7,
                5,
            ],
            "label_numeric": [
                1,
                2,
                timedelta(seconds=3),
                pd.Timedelta("18:58:28"),
                None,
                pd.Timestamp("2022-01-01"),
                4,
                pd.Timedelta("18:40:22"),
                timedelta(seconds=120),
                7,
                5,
            ],
            "answer_temporal_unit": [
                "days",
                "months",
                "seconds",
                "seconds",
                "seconds",
                "date",
                "date",
                "minutes",
                "minutes",
                "years",
                "foo",
            ],
        }
    )
    expected_df = test_df_tot.copy().assign(
        error=[
            0,
            0,
            timedelta(0),
            pd.Timedelta("-1 days +23:50:00"),
            None,
            pd.Timedelta("1 days +00:00:00"),
            0,
            pd.Timedelta("-1 days +23:58:00"),
            timedelta(0),
            0,
            0,
        ],
        error_numeric=[0.0, 0.0, 0.0, -600.0, np.nan, 1, np.nan, -2, 0.0, 0.0, np.nan],
        response_digit=[
            1,
            2,
            3.0,
            pd.Timedelta("18:48:28").total_seconds(),
            np.nan,
            pd.Timestamp("2022-01-02 00:00:00").timestamp(),
            np.nan,
            pd.Timedelta("18:38:22").total_seconds() / 60,
            2.0,
            7,
            np.nan,
        ],
        label_digit=[
            1,
            2,
            3.0,
            pd.Timedelta("18:58:28").total_seconds(),
            np.nan,
            pd.Timestamp("2022-01-01 00:00:00").timestamp(),
            np.nan,
            pd.Timedelta("18:40:22").total_seconds() / 60,
            2.0,
            7,
            np.nan,
        ],
    )
    pd.testing.assert_frame_equal(tot_measure_error(test_df_tot), expected_df)


def test_ttqa_error_measurement():
    test_df_ttqa = pd.DataFrame(
        {
            "response_numeric": [1, 2, 3, 1999, pd.Timestamp("2022-01-01")],
            "label_numeric": [1, 2, 3, 1998, pd.Timestamp("2022-01-02")],
            "answer_temporal_unit": ["years", "months", "days", "date_years", "date"],
        }
    )
    expected_df = test_df_ttqa.copy().assign(
        error=[0, 0, 0, 1, pd.Timedelta("-1 days +00:00:00")],
        error_numeric=[0, 0, 0, 1, -1],
        response_digit=[1, 2, 3, 1999, pd.Timestamp("2022-01-01 00:00:00").timestamp()],
        label_digit=[1, 2, 3, 1998, pd.Timestamp("2022-01-02 00:00:00").timestamp()],
    )
    pd.testing.assert_frame_equal(ttqa_measure_error(test_df_ttqa), expected_df)
