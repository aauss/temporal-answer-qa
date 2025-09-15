import os

from accelerate.test_utils.testing import get_backend
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
ACCESS_TOKEN = os.environ["HF_TOKEN"]
DEVICE, _, _ = get_backend()


class HFModel:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            token=ACCESS_TOKEN,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            token=ACCESS_TOKEN,
        )

    def generate_with_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool,
        continue_final_message: bool,
        max_new_tokens: int,
    ) -> str:
        chat = self.tokenizer.apply_chat_template(
            messages,
            tokenizer=True,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            return_tensors="pt",
            return_dict=True,
        ).to(DEVICE)
        chat_length = chat["input_ids"].shape[1]
        outputs = self.model.generate(
            **chat,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response_str = self.tokenizer.decode(outputs[0][chat_length:], skip_special_tokens=True)
        return response_str
