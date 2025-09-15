import json
from pathlib import Path

from temp_answer_qa import DATA_DIR, ToTSplit, TTQASplit, LastToken, Prompting


def load_chat_template(template_path: str | Path) -> list[dict[str, str]]:
    with open(template_path, "r") as f:
        return json.load(f)["messages"]


class ToTChatBuilder:
    """Chat builder for Test of Time (ToT) datasets."""

    def __init__(self, prompting: Prompting, split: ToTSplit | None = None):
        self.prompting = prompting
        self.split = split

    def build_chat(
        self, question: str, last_token: LastToken, instruction: str
    ) -> list[dict[str, str]]:
        conversation = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": question},
        ]

        if self.prompting == Prompting.few_shot:
            conversation = self._add_few_shot_examples(conversation)

        if last_token == LastToken.continue_final_message:
            conversation.append({"role": "assistant", "content": 'JSON = {"explanation":'})

        return conversation

    def _add_few_shot_examples(self, conversation: list[dict[str, str]]) -> list[dict[str, str]]:
        """Add few-shot examples for ToT dataset."""
        few_shot_path = DATA_DIR / f"prompts/tot_{self.split}_few_shot.json"
        return load_chat_template(few_shot_path) + conversation


class TTQAChatBuilder:
    """Chat builder for the TempTabQA dataset."""

    def __init__(self, prompting: Prompting, split: TTQASplit | None = None):
        self.prompting = prompting
        self.split = split
        self.system_prompt = (DATA_DIR / "prompts/ttqa_system_prompt_zero_shot.txt").read_text()

    def build_chat(self, question: str, table_context: str) -> list[dict[str, str]]:
        question_with_context = f"Table:\n\n{table_context}\n{question}\n\n\nA: "

        if self.prompting == Prompting.zero_shot:
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question_with_context + "Let’s think step by step. "},
            ]
        elif self.prompting == Prompting.few_shot:
            return self._add_few_shot_examples([{"role": "user", "content": question_with_context}])
        else:
            raise ValueError(f"Unknown prompting: {self.prompting}")

    def _add_few_shot_examples(self, conversation: list[dict[str, str]]) -> list[dict[str, str]]:
        few_shot_path = DATA_DIR / "prompts/ttqa_few_shot.json"
        return load_chat_template(few_shot_path) + conversation
