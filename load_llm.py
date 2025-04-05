import torch
from transformers import AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from config import BASE_MODEL
from config import CUSTOM_PATH
class LLMLoader:
    def __init__(self, system_prompt, query_wrapper_prompt):
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            local_files_only=True,           # ✅ 로컬에서만 불러오게 설정
            trust_remote_code=True,           # ✅ 일부 모델은 필수
            token=False,
            cache_dir=CUSTOM_PATH
        )
        self.stopping_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.system_prompt = system_prompt
        self.query_wrapper_prompt = query_wrapper_prompt
        self.base_model = BASE_MODEL

    def load_llm(self):
        print("Loading LLM...")

        return HuggingFaceLLM(
            context_window=8192,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.7, "do_sample": False},
            system_prompt=self.system_prompt,
            query_wrapper_prompt=self.query_wrapper_prompt,
            tokenizer_name=self.base_model,
            model_name=self.base_model,
            device_map="auto",
            stopping_ids=self.stopping_ids,
            tokenizer_kwargs={
                "max_length": 4096,
                "local_files_only": True,
                "trust_remote_code": True,
                "cache_dir":CUSTOM_PATH
            },
            model_kwargs={
                "torch_dtype": torch.float16,
                "local_files_only": True,
                "trust_remote_code": True,
                "cache_dir":CUSTOM_PATH
            }
        )
