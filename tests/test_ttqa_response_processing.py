import datetime

import numpy as np
import pytest

from temp_answer_qa.response_processing import TTQAResponseParser, TTQAResponseToNumericObj

MODEL_RESPONSES = [
    "Adolf von Baeyer was born on 31 October 1835. He married Adelheid Bendemann in 1868. To find out his age at the time of marriage, subtract his birth year from the year of marriage: 1868-1835=33.\nFinal Answer: 33 years",
    "Jeffrey Epstein was apprehended on July 6, 2019. He died by suicide on August 10, 2019. To find the number of months between these two dates, first, find the difference in years: 8 months (August) - 7 months (July) = 1 month. Then, find the difference in the remaining days: 10 - 6 = 4 days. Convert the 4 days to months by dividing by 30 (approximate number of days in a month): 4/30 = 0.133 months. Since 0.133 is less than 1, it is negligible and can be ignored. Now, convert 1 year to months: 1 year = 12 months. Now, add the number of years and the number of months to the remaining days' equivalent in months: 1 year + 0 months + 0.133 months = 12 + 0 + 0.133 = 12.133 months. 0.133 months is negligible and can be ignored. Therefore, Epstein died approximately 12 months after being apprehended.\nFinal Answer: 12 months",
    "To find the year with the highest GDP growth, we need to compare the GDP growth rates for 2018, 2019, 2020, and 2021. \n\nAccording to the table, the GDP growth rates are as follows:\n- 2018: 3.1%\n- 2019e: 1.2%\n- 2020: -0.9%\n- 2021f: 3.3%\n\nThe highest GDP growth rate is 3.3%, which is for 2021. \nFinal Answer: 2021",
    "Ingenuity was deployed on April 3, 2021. The picture was taken on April 7, 2021. To find the number of days between these two dates, subtract the deployment date from the picture date: \n\nApril 7, 2021 - April 3, 2021 = 4 days\n\nThe picture was taken 4 days after Ingenuity was deployed.\nFinal Answer: 4 days",
    "Pierre Auguste Cot was born on 17 February 1837.\nFinal Answer: 17 February 1837",
    "Albert A. Michelson was born on December 19, 1852, and he died on May 9, 1931. To find the age at which he died, subtract the birth year from the death year, then subtract the birth month from the death month, and finally subtract the birth day from the death day: 1931 - 1852 = 79; 5 - 12 = -7; 9 - 19 = -10. However, because you cannot be negative, we simply take 79 years and add 7 months and 10 days to his birth date. Since a month is at least 28 days, it is more accurate to say that Michelson lived 79 years and 7 months. The 7 months is equivalent to 7*30 + 7 = 217 days. Adding 10 days to 217 days results in 227 days, which is equivalent to 7 months and 14 days. Therefore, 7 months and 14 days is more accurate.",
]
ANSWER_FORMATS = [
    "<num_years>",
    "<num_months>",
    "yyyy",
    "<num_days>",
    "%B %d, %Y",
    "%B %d, %Y",
    "<num_years>",
]
EXTRACTED_RESPONSE = [
    "33",
    "12",
    "2021",
    "4",
    datetime.datetime(1837, 2, 17, 0, 0),
    None,
]
EXPECTED_NUMERIC_ANSWERS = [
    33,
    12,
    2021,
    4,
    datetime.datetime(1837, 2, 17, 0, 0),
    None,
]


@pytest.mark.parametrize(
    "model_response,answer_format,expected_answer",
    [
        (MODEL_RESPONSES[0], ANSWER_FORMATS[0], EXTRACTED_RESPONSE[0]),
        (MODEL_RESPONSES[1], ANSWER_FORMATS[1], EXTRACTED_RESPONSE[1]),
        (MODEL_RESPONSES[2], ANSWER_FORMATS[2], EXTRACTED_RESPONSE[2]),
        (MODEL_RESPONSES[3], ANSWER_FORMATS[3], EXTRACTED_RESPONSE[3]),
        (MODEL_RESPONSES[4], ANSWER_FORMATS[4], EXTRACTED_RESPONSE[4]),
        (MODEL_RESPONSES[5], ANSWER_FORMATS[5], EXTRACTED_RESPONSE[5]),
    ],
)
def test_extract_response(model_response, answer_format, expected_answer):
    ttqa_response_parser = TTQAResponseParser()
    print(model_response, answer_format, expected_answer)
    assert ttqa_response_parser.extract_response(model_response, answer_format) == expected_answer


@pytest.mark.parametrize(
    "extracted_answer,answer_format,expected_numeric_answer",
    [
        (EXTRACTED_RESPONSE[0], ANSWER_FORMATS[0], EXPECTED_NUMERIC_ANSWERS[0]),
        (EXTRACTED_RESPONSE[1], ANSWER_FORMATS[1], EXPECTED_NUMERIC_ANSWERS[1]),
        (EXTRACTED_RESPONSE[2], ANSWER_FORMATS[2], EXPECTED_NUMERIC_ANSWERS[2]),
        (EXTRACTED_RESPONSE[3], ANSWER_FORMATS[3], EXPECTED_NUMERIC_ANSWERS[3]),
        (EXTRACTED_RESPONSE[4], ANSWER_FORMATS[4], EXPECTED_NUMERIC_ANSWERS[4]),
        (EXTRACTED_RESPONSE[5], ANSWER_FORMATS[5], EXPECTED_NUMERIC_ANSWERS[5]),
    ],
)
def test_cast_answer_to_numeric(extracted_answer, answer_format, expected_numeric_answer):
    ttqa_response_to_numeric_obj = TTQAResponseToNumericObj()
    assert (
        ttqa_response_to_numeric_obj.cast_response_to_numeric(extracted_answer, answer_format)
        == expected_numeric_answer
    )
    


def test_wrong_input_parser():
    ttqa_responsed_parser = TTQAResponseParser()
    assert ttqa_responsed_parser.extract_response("bla", np.nan) is None
    pytest.raises(ValueError, ttqa_responsed_parser.extract_response, "bla", "some_format")
    assert ttqa_responsed_parser.extract_response("finall answer: 192", "<num_days>") is None
    assert ttqa_responsed_parser.extract_response("finall answer: 2022-06-06", "%B %d, %Y") is None
    assert ttqa_responsed_parser.extract_response("finall answer: 1999", "yyyy") is None
    assert ttqa_responsed_parser.extract_response("Final Answer: 2022-99-99", "%B %d, %Y") is None


def test_wrong_input_resonse_to_numeric():
    ttqa_responsed_parser = TTQAResponseToNumericObj()
    assert ttqa_responsed_parser.cast_response_to_numeric("bla", np.nan) is None
    assert ttqa_responsed_parser.cast_response_to_numeric("bla", "foo") is None
    assert ttqa_responsed_parser.cast_response_to_numeric("2999/01/01", "%B %d, %Y") is None
