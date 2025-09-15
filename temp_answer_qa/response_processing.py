import ast
import json
import re
import warnings
from datetime import timedelta

import datefinder
import pandas as pd

from temp_answer_qa import LastToken


def tot_process_response(responses: pd.DataFrame, last_token: LastToken) -> pd.DataFrame:
    tot_json_parser = ToTJSONParser(last_token=last_token)
    tot_resp2num = ToTResponseToNumericObj()
    return responses.assign(
        # String response to JSON
        response_json=lambda df: df["response"].apply(tot_json_parser.model_response_to_json),
        response_json_wo_explanation=lambda df: df["response_json"].apply(
            tot_json_parser.remove_explanation_from_json
        ),
        label_json=lambda df: df["label"].apply(ast.literal_eval),
        # JSON response to time-aware numeric objects
        response_numeric=lambda df: df["response_json_wo_explanation"].apply(
            tot_resp2num.cast_response_to_numeric
        ),
        label_numeric=lambda df: df["label_json"].apply(tot_resp2num.cast_response_to_numeric),
    )


class ToTJSONParser:
    def __init__(self, last_token: LastToken):
        self.last_token = last_token
        self.json_regex = re.compile(r"({\"explanation\"\s*:.*})", flags=re.DOTALL)

    def label_to_json(self, label: str) -> dict:
        return ast.literal_eval(label)

    def model_response_to_json(self, model_response: str) -> dict | None:
        fixed_model_response = self._fix_model_response(model_response)
        if fixed_model_response:
            return self._try_parse_as_json(fixed_model_response)
        else:
            return None

    def _fix_model_response(self, model_response: str) -> str | None:
        if self.last_token == LastToken.continue_final_message:
            model_response = self._restore_json_in_model_response(model_response)
        model_response = self._replace_newlines(model_response)
        return self._extract_json_from_str(model_response)

    def _restore_json_in_model_response(self, model_response: str) -> str:
        # If we generate with continue_final_message==True, the model response is missing the opening curly brace.
        return ('{"explanation":' + model_response).strip()

    def _replace_newlines(self, model_response: str) -> str:
        # Some models produce newlines which cannot be parsed by json.loads.
        return model_response.replace("\n", " ")

    def _extract_json_from_str(self, model_response: str) -> str | None:
        # Models sometimes return more than just the expected JSON.
        match = self.json_regex.search(model_response)
        if match:
            return match.group(1)
        else:
            return None

    def _try_parse_as_json(self, str: str) -> dict | None:
        try:
            return json.loads(str)
        except (json.JSONDecodeError, TypeError):
            return None

    def remove_explanation_from_json(self, json: dict) -> dict | None:
        if json:
            return {k: v for k, v in json.items() if k != "explanation"}
        else:
            return None


class ToTResponseToNumericObj:
    def cast_response_to_numeric(self, response: dict):
        try:
            if {"age"}.issubset(response.keys()):
                return self._cast_age_response(response["age"])
            elif {"date"}.issubset(response.keys()):
                return self._cast_date_response(response["date"])
            elif {"answer"}.issubset(response.keys()):
                return self._cast_unspecified_response(response["answer"])
            elif (
                {"A", "B", "C"} == response.keys()
                or {"X", "Y", "Z"} == response.keys()
                or {"H", "M", "S"} == response.keys()
                or {"hours", "minutes", "seconds"}.issuperset(response.keys())
            ):
                return self._cast_seconds_response(response)
            elif {"time", "day"}.issubset(response.keys()) or {
                "days",
                "hours",
                "minutes",
                "seconds",
            }.issubset(response.keys()):
                return self._cast_timezone(response)
            else:
                None
        except (pd._libs.tslibs.parsing.DateParseError, ValueError, AttributeError):
            return None

    def _cast_age_response(self, response):
        return int(response)

    def _cast_date_response(self, response):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(response, dayfirst=False)

    def _cast_unspecified_response(self, response):
        try:
            if isinstance(response, int) or str(response).isdigit():
                return int(response)
            elif "BC" in response:
                return -int(response.replace("BC", ""))
            elif "AD" in response:
                return int(response.replace("AD", ""))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    return pd.to_datetime(response, dayfirst=False)
        except TypeError as e:
            return None

    def _cast_seconds_response(self, response: dict):
        target_keys = ["hours", "minutes", "seconds"]
        response_mapped = {target_keys[i]: int(v) for i, (k, v) in enumerate(response.items())}
        return timedelta(**response_mapped)

    def _cast_timezone(self, response: dict):
        if "time" in response.keys():
            target_keys = ["hours", "minutes", "seconds"]
            time = response["time"].split(":")
            time_mapped = {target_keys[i]: int(v) for i, v in enumerate(time)}
            days_mapped = int(
                str(response["day"]).replace("same_day", "0").replace("previous_day", "-1")
            )
            days_time = {"days": days_mapped} | time_mapped
            return timedelta(**days_time)
        else:
            response_mapped = {k: int(v) for k, v in response.items()}
            return timedelta(**response_mapped)


def ttqa_process_response(responses: pd.DataFrame) -> pd.DataFrame:
    ttqa_json_parser = TTQAResponseParser()
    ttqa_resp2num = TTQAResponseToNumericObj()
    return responses.assign(
        response_extracted=lambda df: df.apply(
            lambda row: ttqa_json_parser.extract_response(row["response"], row["answer_format"]),
            axis=1,
        ),
        response_numeric=lambda df: df.apply(
            lambda row: ttqa_resp2num.cast_response_to_numeric(
                row["response_extracted"], row["answer_format"]
            ),
            axis=1,
        ),
        label_numeric=lambda df: df.apply(
            lambda row: ttqa_resp2num.cast_response_to_numeric(row["label"], row["answer_format"]),
            axis=1,
        ),
    )


class TTQAResponseParser:
    def __init__(self):
        self._EXTRACTION_FUNCTIONS = {
            "<num_years>": self._extract_num_years,
            "yyyy": self._extract_yyyy,
            "%B %d, %Y": self._extract_date,
            "<num_days>": self._extract_num_days,
            "<num_months>": self._extract_num_months,
        }

    def extract_response(self, model_response, answer_format):
        if pd.isna(model_response) or pd.isna(answer_format):
            return None
        if answer_format not in self._EXTRACTION_FUNCTIONS.keys():
            raise ValueError(f"No extraction function defined for format: {answer_format}")
        func = self._EXTRACTION_FUNCTIONS.get(answer_format)
        return func(model_response)

    def _try_extract_int_after_final_answer(self, model_response):
        match = re.search(r"Final Answer:.*?(\d+)", model_response)
        if match:
            return match[1]
        else:
            return None

    def _extract_num_years(self, model_response):
        return self._try_extract_int_after_final_answer(model_response)

    def _extract_num_months(self, model_response):
        return self._try_extract_int_after_final_answer(model_response)

    def _extract_num_days(self, model_response):
        return self._try_extract_int_after_final_answer(model_response)

    def _extract_yyyy(self, model_response):
        match = re.search(r"Final Answer:.*?(\d{4})", model_response)
        if match:
            return match[1]
        else:
            return None

    def _extract_date(self, model_response):
        match = re.search("Final Answer:(.*)", model_response)
        if not match:
            return None
        final_answer = match[1]
        try:
            return next(datefinder.find_dates(final_answer))
        except StopIteration:
            return None


class TTQAResponseToNumericObj:
    def cast_response_to_numeric(self, response, answer_format: str):
        if pd.isna(response) or pd.isna(answer_format):
            return None
        try:
            if ("<num_" in answer_format) or (answer_format == "yyyy"):
                return self._cast_numeric_response(response)
            elif answer_format == r"%B %d, %Y":
                if isinstance(response, str):
                    return pd.to_datetime(response)
                else:
                    # Datefinder package already returns datetime object
                    return response
            else:
                return None
        except pd.errors.OutOfBoundsDatetime:
            return None

    def _cast_numeric_response(self, response: str):
        if response:
            try:
                return int(response)
            except ValueError:
                return float(response)
        else:
            return None
