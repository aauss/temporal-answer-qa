# tests/test_inference.py
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from temp_answer_qa import ToTSplit
from temp_answer_qa.inference import Prompting, TTQASplit, LastToken, ttqa, tot


@pytest.mark.parametrize(
    "split,build_chat_values,mock_df",
    [
        (
            TTQASplit.head,
            ("Q1", "T1"),
            pd.DataFrame({"question": ["Q1"], "table_context": ["T1"], "split": ["head"]}),
        ),
        (
            TTQASplit.tail,
            ("Q2", "T2"),
            pd.DataFrame({"question": ["Q2"], "table_context": ["T2"], "split": ["tail"]}),
        ),
    ],
)
@patch("temp_answer_qa.inference.data_loader")
@patch("temp_answer_qa.inference.TTQAChatBuilder")
@patch("temp_answer_qa.inference.HFModel")
def test_ttqa(
    mock_HFModel,
    mock_TTQAChatBuilder,
    mock_data_loader,
    tmp_path,
    split,
    build_chat_values,
    mock_df,
):
    mock_data_loader.load_ttqa.return_value = mock_df

    mock_chat_builder = MagicMock()
    mock_chat_builder.build_chat.return_value = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "response"},
    ]
    mock_TTQAChatBuilder.return_value = mock_chat_builder

    mock_model = MagicMock()
    mock_model.generate_with_chat_template.return_value = "response"
    mock_HFModel.return_value = mock_model

    ttqa(
        prompting=Prompting.zero_shot,
        split=split,
        model_name="test-model",
        last_token=LastToken.add_generation_prompt,
        test_mode=True,
        output_folder=tmp_path,
    )

    # Assertions
    mock_data_loader.load_ttqa.assert_called_once_with(split=split, test_mode=True)
    mock_chat_builder.build_chat.assert_called_once_with(*build_chat_values)
    mock_model.generate_with_chat_template.assert_called_once_with(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "response"},
        ],
        add_generation_prompt=True,
        continue_final_message=False,
        max_new_tokens=256,
    )
    expected_filename = f"ttqa_{split}_test-model_zero-shot_add_generation_prompt.csv"
    assert (tmp_path / expected_filename).exists()


@pytest.mark.parametrize(
    "split,build_chat_values,mock_df",
    [
        (
            ToTSplit.arithmetic,
            ("Q1", "add_generation_prompt", "I1"),
            pd.DataFrame(
                {
                    "question_wo_instruct": ["Q1"],
                    "instruction": ["I1"],
                    "split": ["arithmetic"],
                }
            ),
        ),
        (
            ToTSplit.semantic,
            ("Q2", "add_generation_prompt", "I2"),
            pd.DataFrame(
                {
                    "question_wo_instruct": ["Q2"],
                    "instruction": ["I2"],
                    "split": ["semantic"],
                }
            ),
        ),
    ],
)
@patch("temp_answer_qa.inference.data_loader")
@patch("temp_answer_qa.inference.ToTChatBuilder")
@patch("temp_answer_qa.inference.HFModel")
def test_tot(
    mock_HFModel, mock_ToTChatBuilder, mock_data_loader, tmp_path, split, build_chat_values, mock_df
):
    mock_data_loader.load_tot.return_value = mock_df

    mock_chat_builder = MagicMock()
    mock_chat_builder.build_chat.return_value = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "response"},
    ]
    mock_ToTChatBuilder.return_value = mock_chat_builder

    mock_model = MagicMock()
    mock_model.generate_with_chat_template.return_value = "response"
    mock_HFModel.return_value = mock_model

    tot(
        prompting=Prompting.zero_shot,
        split=split,
        model_name="test-model",
        last_token=LastToken.add_generation_prompt,
        test_mode=True,
        output_folder=tmp_path,
    )

    # Assertions
    mock_data_loader.load_tot.assert_called_once_with(split=split, test_mode=True)
    mock_chat_builder.build_chat.assert_called_once_with(*build_chat_values)
    mock_model.generate_with_chat_template.assert_called_once_with(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "response"},
        ],
        add_generation_prompt=True,
        continue_final_message=False,
        max_new_tokens=512,
    )
    expected_filename = f"tot_{split}_test-model_zero-shot_add_generation_prompt.csv"
    assert (tmp_path / expected_filename).exists()
