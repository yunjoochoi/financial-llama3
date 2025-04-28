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
            context_window=8192, # 한번에 읽을 수 있는 총 토큰 수, 약 6,000~6,500 단어(LLaMA 3-8B이 8192 토큰까지 context를 지원)
            max_new_tokens=512, # 최대 몇 개 토큰 생성할 건지
            generate_kwargs={"temperature": 0.7, "top_p": 0.9, "do_sample": True}, # 확률 상위 top_p 안에서 샘플링, temperature=0.7: 출력 다양성 조정(낮을수록 덜 랜덤)
                                                                                # "do_sample" false로하면 가장 확률 높은 샘플만 선택
            system_prompt=self.system_prompt,
            query_wrapper_prompt=self.query_wrapper_prompt,
            tokenizer_name=self.base_model,
            model_name=self.base_model,
            device_map="auto",
            stopping_ids=self.stopping_ids,
            tokenizer_kwargs={
                "max_length": 8192, # 한 번에 Tokenizer가 처리할 최대 토큰 수 (보통 모델 context_window와 맞춤)
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
