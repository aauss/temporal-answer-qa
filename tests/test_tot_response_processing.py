import pandas as pd

from temp_answer_qa import LastToken
from temp_answer_qa.response_processing import ToTJSONParser, ToTResponseToNumericObj


def test_model_response_to_json():
    model_output = """ "First, we need ...\nThe departure time in UTC is 17:35:53. ... hours.", "day": "same_day", "time": "13:04:13"}  Explanation: 1. Convert departure time from PST to UTC: 09:35:53 + 8 hours = 17:35:53 UTC 2. Add flight duration to departure time in UTC: 17:35:53 + 22:04:10 = 39:39:03 UTC 3. Convert arrival time in UTC to PST: 39:39:03 - 8 hours = 13:39:03 PST  The plane lands in Location B at 13:39:03 PST."""
    expected_output = {
        "explanation": "First, we need ... The departure time in UTC is 17:35:53. ... hours.",
        "day": "same_day",
        "time": "13:04:13",
    }
    tot_json_parser = ToTJSONParser(last_token=LastToken.continue_final_message)
    output = tot_json_parser.model_response_to_json(model_output)
    assert output == expected_output


def test_restore_json_in_model_response():
    model_output = ' "First, we need to find the ... So Natalie\'s age is 316 + 731 + 260 = 1307 days.", "answer": 1307}'
    expected_output = '{"explanation": "First, we need to find the ... So Natalie\'s age is 316 + 731 + 260 = 1307 days.", "answer": 1307}'
    tot_json_parser = ToTJSONParser(last_token=LastToken.continue_final_message)
    output = tot_json_parser._restore_json_in_model_response(model_output)
    assert output == expected_output


def test_replace_newlines():
    model_output = """ "To find the number of days ...\n- The number of days between December 1 to December 27 are: 27 - 1 + 1 = 27 days\n\n...So, the total days is 208 + 118 = 326 days.", "answer": 326}"""
    expected_output = """ "To find the number of days ... - The number of days between December 1 to December 27 are: 27 - 1 + 1 = 27 days  ...So, the total days is 208 + 118 = 326 days.", "answer": 326}"""
    tot_json_parser = ToTJSONParser(last_token=LastToken.continue_final_message)
    output = tot_json_parser._replace_newlines(model_output)
    assert output == expected_output


def test_extract_json_from_str():
    model_output = """{"explanation": "First, we need to convert the departure time from PST to UTC by adding 8 hours. The departure time in UTC is 17:35:53. Then, we add the flight duration to the departure time in UTC to get the arrival time in UTC. Finally, we convert the arrival time in UTC to PST by subtracting 8 hours.", "day": "same_day", "time": "13:04:13"}  Explanation: 1. Convert departure time from PST to UTC: 09:35:53 + 8 hours = 17:35:53 UTC 2. Add flight duration to departure time in UTC: 17:35:53 + 22:04:10 = 39:39:03 UTC 3. Convert arrival time in UTC to PST: 39:39:03 - 8 hours = 13:39:03 PST  The plane lands in Location B at 13:39:03 PST."""
    expected_output = """{"explanation": "First, we need to convert the departure time from PST to UTC by adding 8 hours. The departure time in UTC is 17:35:53. Then, we add the flight duration to the departure time in UTC to get the arrival time in UTC. Finally, we convert the arrival time in UTC to PST by subtracting 8 hours.", "day": "same_day", "time": "13:04:13"}"""
    tot_json_parser = ToTJSONParser(last_token=LastToken.continue_final_message)
    output = tot_json_parser._extract_json_from_str(model_output)
    assert output == expected_output


def test_answer_to_numeric():
    tot_resp2num = ToTResponseToNumericObj()
    assert tot_resp2num.cast_response_to_numeric({"age": "7"}) == 7
    assert tot_resp2num.cast_response_to_numeric({"age": 7}) == 7
    assert tot_resp2num.cast_response_to_numeric({"answer": 7}) == 7
    assert tot_resp2num.cast_response_to_numeric({"answer": "7"}) == 7
    assert tot_resp2num.cast_response_to_numeric({"answer": "7 BC"}) == -7
    assert tot_resp2num.cast_response_to_numeric({"answer": "7 AD"}) == 7
    assert tot_resp2num.cast_response_to_numeric({"answer": "10/01/2015"}) == pd.to_datetime(
        "10/01/2015"
    )
    assert tot_resp2num.cast_response_to_numeric({"date": "10/01/2015"}) == pd.to_datetime(
        "10/01/2015"
    )
    assert tot_resp2num.cast_response_to_numeric(
        {"hours": 17, "minutes": 48, "seconds": 51}
    ) == pd.Timedelta(hours=17, minutes=48, seconds=51)
    assert tot_resp2num.cast_response_to_numeric(
        {"days": 2, "hours": 17, "minutes": 48, "seconds": 51}
    ) == pd.Timedelta(days=2, hours=17, minutes=48, seconds=51)
    assert tot_resp2num.cast_response_to_numeric({"hours": 17, "minutes": 48}) == pd.Timedelta(
        hours=17, minutes=48
    )
    assert tot_resp2num.cast_response_to_numeric({"H": 17, "M": 48, "S": 51}) == pd.Timedelta(
        hours=17, minutes=48, seconds=51
    )
    assert tot_resp2num.cast_response_to_numeric({"A": 17, "B": 48, "C": 51}) == pd.Timedelta(
        hours=17, minutes=48, seconds=51
    )
    assert tot_resp2num.cast_response_to_numeric({"X": 17, "Y": 48, "Z": 51}) == pd.Timedelta(
        hours=17, minutes=48, seconds=51
    )
    assert tot_resp2num.cast_response_to_numeric({"day": 1, "time": "21:45:12"}) == pd.Timedelta(
        days=1, hours=21, minutes=45, seconds=12
    )
    assert tot_resp2num.cast_response_to_numeric(
        {"day": "previous_day", "time": "21:45:12"}
    ) == pd.Timedelta(days=-1, hours=21, minutes=45, seconds=12)
    assert tot_resp2num.cast_response_to_numeric(
        {"day": "same_day", "time": "21:45:12"}
    ) == pd.Timedelta(hours=21, minutes=45, seconds=12)


def test_wrong_input_response_to_numeric():
    tot_resp2num = ToTResponseToNumericObj()
    assert tot_resp2num.cast_response_to_numeric({"foo": 42}) is None
    assert tot_resp2num.cast_response_to_numeric({"date": "999-999"}) is None
    assert tot_resp2num.cast_response_to_numeric({"answer": " - 2001 AD"}) is None
    assert tot_resp2num.cast_response_to_numeric({"answer": 1.45}) is None
    assert tot_resp2num.cast_response_to_numeric({"answer": {"foo": "bar"}}) is None


def test_label_to_json():
    tot_json_parser = ToTJSONParser(last_token=LastToken.continue_final_message)
    assert tot_json_parser.label_to_json('{"foo": 42}') == {"foo": 42}


def test_wrong_input_model_response_to_json():
    tot_json_parser = ToTJSONParser(last_token=LastToken.continue_final_message)
    assert tot_json_parser.model_response_to_json("bla") is None
    assert tot_json_parser.model_response_to_json('{"foo": 01 }') is None


def test_remove_explanation_from_json():
    tot_json_parser = ToTJSONParser(last_token=LastToken.continue_final_message)
    assert tot_json_parser.remove_explanation_from_json(None) is None
    assert tot_json_parser.remove_explanation_from_json(
        {"explanation": "abc", "answer": {"foo": "bar"}}
    ) == {"answer": {"foo": "bar"}}
