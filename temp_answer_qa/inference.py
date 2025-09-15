from pathlib import Path

from tqdm import tqdm

from temp_answer_qa import LastToken, Prompting, ToTSplit, TTQASplit
from temp_answer_qa.chat_builder import ToTChatBuilder, TTQAChatBuilder
from temp_answer_qa.data_loader import DataLoader
from temp_answer_qa.models import HFModel

data_loader = DataLoader()


def ttqa(
    prompting: Prompting,
    split: TTQASplit,
    model_name: str,
    last_token: LastToken,
    output_folder: Path,
    test_mode: bool = False,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    dataset = data_loader.load_ttqa(split=split, test_mode=test_mode)
    chat_builder = TTQAChatBuilder(prompting=prompting, split=split)
    hf_model = HFModel(model_name=model_name)
    responses = []
    for _, row in tqdm(
        dataset.iterrows(),
        desc=f"Inference on TTQA with model: {model_name}, prompting: {prompting}, split: {split}, last token: {last_token}",
        total=dataset.shape[0]
    ):
        question = row["question"]
        table_context = row["table_context"]
        chat = chat_builder.build_chat(question, table_context)
        response = hf_model.generate_with_chat_template(
            chat,
            add_generation_prompt=(last_token == LastToken.add_generation_prompt),
            continue_final_message=(last_token == LastToken.continue_final_message),
            max_new_tokens=256,
        )
        responses.append(response)
    dataset.loc[:, "response"] = responses
    output_path = (
        output_folder / f"ttqa_{split}_{model_name.split('/')[-1]}_{prompting}_{last_token}.csv"
    )
    dataset.to_csv(output_path, index=False)


def tot(
    prompting: Prompting,
    split: ToTSplit,
    model_name: str,
    last_token: LastToken,
    output_folder: Path,
    test_mode: bool = False,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    dataset = data_loader.load_tot(split=split, test_mode=test_mode)
    chat_builder = ToTChatBuilder(prompting=prompting, split=split)
    hf_model = HFModel(model_name=model_name)
    responses = []
    for _, row in tqdm(
        dataset.iterrows(),
        desc=f"Inference on ToT with model: {model_name}, prompting: {prompting}, split: {split}, last token: {last_token}",
        total=dataset.shape[0]
    ):
        question = row["question_wo_instruct"]
        instruction = row["instruction"]
        chat = chat_builder.build_chat(question, last_token, instruction)
        response = hf_model.generate_with_chat_template(
            chat,
            add_generation_prompt=(last_token == LastToken.add_generation_prompt),
            continue_final_message=(last_token == LastToken.continue_final_message),
            max_new_tokens=512,
        )
        responses.append(response)
    dataset.loc[:, "response"] = responses
    output_path = (
        output_folder / f"tot_{split}_{model_name.split('/')[-1]}_{prompting}_{last_token}.csv"
    )
    dataset.to_csv(output_path, index=False)
