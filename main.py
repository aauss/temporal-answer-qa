from pathlib import Path

import typer

from temp_answer_qa import (
    LastToken,
    Prompting,
    ToTSplit,
    TTQASplit,
)
from temp_answer_qa.evaluate import eval_tot, eval_ttqa
from temp_answer_qa.inference import tot, ttqa

app = typer.Typer()
RESPONSE_DIR = Path(__file__).parent / "data/responses/"
EVAL_DIR = Path(__file__).parent / "data/responses_evaluated/"


@app.command()
def inference_tot(
    model_name: str,
    last_token: LastToken,
    prompting: Prompting,
    split: ToTSplit,
    test_mode: bool = False,
    output_folder: Path = RESPONSE_DIR,
):
    tot(prompting, split, model_name, last_token, output_folder, test_mode)


@app.command()
def inference_ttqa(
    model_name: str,
    last_token: LastToken,
    prompting: Prompting,
    split: TTQASplit,
    test_mode: bool = False,
    output_folder: Path = RESPONSE_DIR,
):
    ttqa(prompting, split, model_name, last_token, output_folder, test_mode)


@app.command()
def evaluate_tot(results_folder: Path, last_token: LastToken, output_folder: Path = EVAL_DIR):
    eval_tot(results_folder, last_token, output_folder)


@app.command()
def evaluate_ttqa(results_folder: Path, last_token: LastToken, output_folder: Path = EVAL_DIR):
    eval_ttqa(results_folder, last_token, output_folder)


if __name__ == "__main__":
    app()
