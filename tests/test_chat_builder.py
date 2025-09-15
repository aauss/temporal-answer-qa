from pathlib import Path

import pytest

from temp_answer_qa import LastToken, Prompting, ToTSplit
from temp_answer_qa.chat_builder import (
    ToTChatBuilder,
    TTQAChatBuilder,
    load_chat_template,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def test_tot_chat_builder_zero_shot():
    builder = ToTChatBuilder(prompting=Prompting.zero_shot, split=ToTSplit.semantic)
    chat_1 = builder.build_chat(
        question="Natalie and Chris were born on 2004-Feb-18 and 2004-Dec-30 respectively. When Chris was 991 days old, how old was Natalie in days?",
        instruction='Return your answer as a JSON like: JSON = {"explanation": <your step by step solution>, "answer": <num_days>}. Do not include the units in the JSON value.',
        last_token=LastToken.continue_final_message,
    )
    expected_chat_1 = [
        {
            "role": "system",
            "content": 'Return your answer as a JSON like: JSON = {"explanation": <your step by step solution>, "answer": <num_days>}. Do not include the units in the JSON value.',
        },
        {
            "role": "user",
            "content": "Natalie and Chris were born on 2004-Feb-18 and 2004-Dec-30 respectively. When Chris was 991 days old, how old was Natalie in days?",
        },
        {"role": "assistant", "content": 'JSON = {"explanation":'},
    ]
    assert chat_1 == expected_chat_1

    chat_2 = builder.build_chat(
        question="Natalie and Chris were born on 2004-Feb-18 and 2004-Dec-30 respectively. When Chris was 991 days old, how old was Natalie in days?",
        instruction='Return your answer as a JSON like: JSON = {"explanation": <your step by step solution>, "answer": <num_days>}. Do not include the units in the JSON value.',
        last_token=LastToken.add_generation_prompt,
    )
    expected_chat_2 = [
        {
            "role": "system",
            "content": 'Return your answer as a JSON like: JSON = {"explanation": <your step by step solution>, "answer": <num_days>}. Do not include the units in the JSON value.',
        },
        {
            "role": "user",
            "content": "Natalie and Chris were born on 2004-Feb-18 and 2004-Dec-30 respectively. When Chris was 991 days old, how old was Natalie in days?",
        },
    ]
    assert chat_2 == expected_chat_2


def test_tot_chat_builder_few_shot():
    builder = ToTChatBuilder(prompting=Prompting.few_shot, split=ToTSplit.arithmetic)
    chat_1 = builder.build_chat(
        question="Natalie and Chris were born on 2004-Feb-18 and 2004-Dec-30 respectively. When Chris was 991 days old, how old was Natalie in days?",
        instruction='Return your answer as a JSON like: JSON = {"explanation": <your step by step solution>, "answer": <num_days>}. Do not include the units in the JSON value.',
        last_token=LastToken.continue_final_message,
    )
    few_shot_path = DATA_DIR / "prompts/tot_arithmetic_few_shot.json"
    few_shot_examples = load_chat_template(few_shot_path)
    expected_chat_1 = few_shot_examples + [
        {
            "role": "system",
            "content": 'Return your answer as a JSON like: JSON = {"explanation": <your step by step solution>, "answer": <num_days>}. Do not include the units in the JSON value.',
        },
        {
            "role": "user",
            "content": "Natalie and Chris were born on 2004-Feb-18 and 2004-Dec-30 respectively. When Chris was 991 days old, how old was Natalie in days?",
        },
        {"role": "assistant", "content": 'JSON = {"explanation":'},
    ]
    assert chat_1 == expected_chat_1

    builder = ToTChatBuilder(prompting=Prompting.few_shot, split=ToTSplit.semantic)
    chat_2 = builder.build_chat(
        question="Natalie and Chris were born on 2004-Feb-18 and 2004-Dec-30 respectively. When Chris was 991 days old, how old was Natalie in days?",
        instruction='Return your answer as a JSON like: JSON = {"explanation": <your step by step solution>, "answer": <num_days>}. Do not include the units in the JSON value.',
        last_token=LastToken.continue_final_message,
    )
    few_shot_path = DATA_DIR / "prompts/tot_semantic_few_shot.json"
    few_shot_examples = load_chat_template(few_shot_path)
    expected_chat_2 = few_shot_examples + [
        {
            "role": "system",
            "content": 'Return your answer as a JSON like: JSON = {"explanation": <your step by step solution>, "answer": <num_days>}. Do not include the units in the JSON value.',
        },
        {
            "role": "user",
            "content": "Natalie and Chris were born on 2004-Feb-18 and 2004-Dec-30 respectively. When Chris was 991 days old, how old was Natalie in days?",
        },
        {"role": "assistant", "content": 'JSON = {"explanation":'},
    ]
    assert chat_2 == expected_chat_2


def test_ttqa_chat_builder_zero_shot(shared_datadir):
    builder = TTQAChatBuilder(prompting=Prompting.zero_shot)
    table = (shared_datadir / "ttqa_table_formatted.txt").read_text()

    chat_1 = builder.build_chat(
        question="At what age can an Australian both serve and be deployed in the Australian Defence Force?",
        table_context=table,
    )
    expected_chat_1 = [
        {
            "role": "system",
            "content": 'Given an entity-centric table and corresponding question, answer the question by providing step-by-step reasoning and then clearly and concisely stating the final answer using "Final Answer:".\n\nEach table-question pair is presented as a table (identified by "Table:") followed by a question (identified by "Q:"). Tables are presented in a linear format, with columns separated by tabs, rows separated by newlines, and subsections separated by double newlines. If necessary, assume the current date is December, 2022.',
        },
        {
            "role": "user",
            "content": "Table:\n\ncategory\narmy\n\nAustralian Defence Force\nFounded\t1 January 1901 |  ( | 1901-01-01 | ) | (federation)\nCurrent form\t9 February 1976 |  ( | 1976-02-09 | )\nService branches\tRoyal Australian Navy | Australian Army | Royal Australian Air Force\nHeadquarters\tCanberra, ACT, as part of the Australian Defence Organisation\n['Australian Defence Force tri-service flag.jpg | The Australian Defence Force tri-service flag']\n\nLeadership\nCommander-in-Chief\tGeneral David Hurley\nPrime Minister\tAnthony Albanese\nMinister for Defence\tRichard Marles\nChief of the Defence Force\tGeneral Angus Campbell\n\nPersonnel\nMilitary age\t16.5 years (for selection) | 17 years (to serve) | 18 years (to deploy) | 19 years (for special forces deployment)\nConscription\tNo, abolished\nActive personnel\t59,095 (30 June 2020)\nReserve personnel\t28,878 (30 June 2020)\nDeployed personnel\t1,841 (July 2020)\n\nExpenditures\nBudget\tA$ | 44.6 billion (2021-22)\nPercent of GDP\t2.0\n\nIndustry\nDomestic suppliers\tDefence industry of Australia\nAnnual exports\tAround | A$ | 2 billion (2018)\n\nRelated articles\nHistory\tMilitary history of Australia\nRanks\tAustralian Defence Force ranks and insignia\n\n\nAt what age can an Australian both serve and be deployed in the Australian Defence Force?\n\n\nA: Let’s think step by step. ",
        },
    ]
    assert chat_1 == expected_chat_1


def test_ttqa_chat_builder_few_shot(shared_datadir):
    builder = TTQAChatBuilder(prompting=Prompting.few_shot)
    table = (shared_datadir / "ttqa_table_formatted.txt").read_text()

    chat_1 = builder.build_chat(
        question="At what age can an Australian both serve and be deployed in the Australian Defence Force?",
        table_context=table,
    )
    expected_chat_1 = [
        {
            "role": "system",
            "content": 'Given an entity-centric table and corresponding question, answer the question by providing step-by-step reasoning and then clearly and concisely stating the final answer using "Final Answer:".\n\nEach table-question pair is presented as a table (identified by "Table:") followed by a question (identified by "Q:"). Tables are presented in a linear format, with columns separated by tabs, rows separated by newlines, and subsections separated by double newlines. If necessary, assume the current date is December, 2022.\n\nHere is an example that follows these instructions. Answer the provided questions in a similar format:\n',
        },
        {
            "role": "user",
            "content": "Table:\n\nTitle\tBette Davis\nDied\tOctober 6, 1989 | (1989-10-06) | (aged 81) | Neuilly-sur-Seine, France\nResting place\tForest Lawn Memorial Park\nOccupation\tActress\nYears active\t1929-1989\nPolitical party\tDemocratic\nSpouse(s)\tHarmon Oscar Nelson | ( | married |  1932; | divorced |  1938) | Arthur Farnsworth | ( | married |  1940; died 1943) | William Grant Sherry | ( | married |  1945; | divorced |  1950) | Gary Merrill | ( | married |  1950; | divorced |  1960)\nChildren\t3, including B. D. Hyman\nBette Davis\t['BETTEDavis (cropped).jpg | Publicity photo, 1935']\n\n\nQ: How long after Davis's career began did she take the photo?\n\n\nA:",
        },
        {
            "role": "assistant",
            "content": "Davis's career ranged from 1929 to 1989, so her career began in 1929. The photo was taken in 1935. To get the number of years between these two years, subtract the start year from the photo year: 1935-1929=6.\nFinal Answer: 6 years",
        },
        {
            "role": "user",
            "content": 'Table:\n\nTitle\t"2 Hearts 1 Love"\n"2 Hearts 1 Love"\t[\'2 Hearts 1 Love.jpeg\']\n\nSingle by 911\n\nfrom the album | Illuminate... (The Hits and More)\nReleased\t8 September 2013\nRecorded\t2013 | Central Sounds Studio, | Glasgow, Scotland\nGenre\tPop\nSongwriter(s)\t911\nProducer(s)\tJohn McLaughlin\n\n911 singles chronology\n911 singles chronology\t[\'"Wonderland" | (1999) | " | 2 Hearts 1 Love | " | (2013) | "I Wish It Could Be Christmas Everyday" | (2013)\']\n\n\nQ: What year was 2 Hearts 1 Love released?\n\n\nA:',
        },
        {
            "role": "assistant",
            "content": "2 Hearts 1 Love was released on 8 September 2013, which is in the year 2013.\nFinal Answer: 2013",
        },
        {
            "role": "user",
            "content": "Table:\n\nTitle\tDwight Bernard\nDwight Bernard\t['Dwight Bernard.jpg']\n\nPitcher\n\nBorn | ( | 1952-05-31 | ) | May 31, 1952 | (age 70) | Mount Vernon, Illinois\n\nBatted | Right | Threw | Right\n\nMLB debut\nMLB debut\t['June 29, 1978, for the New York Mets']\n\nLast MLB appearance\nLast MLB appearance\t['October 2, 1982, for the Milwaukee Brewers']\n\nMLB statistics\nWin-loss record\t4-8\nEarned run average\t4.14\nStrikeouts\t92\n\nTeams\nTeams\t['New York Mets (1978-1979) | Milwaukee Brewers (1981-1982)']\n\n\nQ: When did Dwight Bernard start playing with the last team he was associated with?\n\n\nA:",
        },
        {
            "role": "assistant",
            "content": "Dwight Bernard has played with the New York Mets and Milwaukee Brewers. Bernard played for the Mets from 1978 to 1979 and the Brewers ,from 1981 to 1982. He was associated with the Brewers in 1982, which is more recent than the Mets in 1979; therefore, the last team he was associated with is the Brewers. He started playing for the Brewers in 1981.\nFinal Answer: 1981",
        },
        {
            "role": "user",
            "content": "Table:\n\ncategory\narmy\n\nAustralian Defence Force\nFounded\t1 January 1901 |  ( | 1901-01-01 | ) | (federation)\nCurrent form\t9 February 1976 |  ( | 1976-02-09 | )\nService branches\tRoyal Australian Navy | Australian Army | Royal Australian Air Force\nHeadquarters\tCanberra, ACT, as part of the Australian Defence Organisation\n['Australian Defence Force tri-service flag.jpg | The Australian Defence Force tri-service flag']\n\nLeadership\nCommander-in-Chief\tGeneral David Hurley\nPrime Minister\tAnthony Albanese\nMinister for Defence\tRichard Marles\nChief of the Defence Force\tGeneral Angus Campbell\n\nPersonnel\nMilitary age\t16.5 years (for selection) | 17 years (to serve) | 18 years (to deploy) | 19 years (for special forces deployment)\nConscription\tNo, abolished\nActive personnel\t59,095 (30 June 2020)\nReserve personnel\t28,878 (30 June 2020)\nDeployed personnel\t1,841 (July 2020)\n\nExpenditures\nBudget\tA$ | 44.6 billion (2021-22)\nPercent of GDP\t2.0\n\nIndustry\nDomestic suppliers\tDefence industry of Australia\nAnnual exports\tAround | A$ | 2 billion (2018)\n\nRelated articles\nHistory\tMilitary history of Australia\nRanks\tAustralian Defence Force ranks and insignia\n\n\nAt what age can an Australian both serve and be deployed in the Australian Defence Force?\n\n\nA: ",
        },
    ]

    assert chat_1 == expected_chat_1


def test_wrong_prompting():
    builder = TTQAChatBuilder(prompting="bla_shot")
    with pytest.raises(ValueError):
        builder.build_chat(
            question="some question",
            table_context="some table",
        )
