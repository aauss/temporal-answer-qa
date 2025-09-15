from enum import StrEnum
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


class TTQASplit(StrEnum):
    head = "head"
    tail = "tail"


class ToTSplit(StrEnum):
    arithmetic = "arithmetic"
    semantic = "semantic"


class LastToken(StrEnum):
    add_generation_prompt = "add_generation_prompt"
    continue_final_message = "continue_final_message"


class Prompting(StrEnum):
    few_shot = "few-shot"
    zero_shot = "zero-shot"
