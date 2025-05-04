# ✅ 개선된 KG 기반 RAG 파이프라인: temp.py

from llama_index.core.postprocessor import SentenceTransformerRerank
from hugging import HuggingFaceNotebookLogin
from dataset import DataSetter
from embdding_setter import EmbedSettings
from load_llm import LLMLoader
from prompt_template import PromptTemplates
from config import ENCODER_MODEL
from KG_graph import KGQueryEngine
from chat_manager import KGChatEngine
import time

# 로그인 및 설정
HuggingFaceNotebookLogin().login()

# 데이터 로드
documents = DataSetter().load_dataset()

# 프롬프트 준비
promptTP = PromptTemplates()
system_prompt = promptTP.get_system_prompt()
query_wrapper_prompt = promptTP.get_query_wrapper_prompt()

# 임베딩 및 LLM 설정
settting = EmbedSettings().set_and_get_llama_settings()
settting.llm = LLMLoader(system_prompt, query_wrapper_prompt).load_llm()
# Reranker 준비
rerank = SentenceTransformerRerank(model=ENCODER_MODEL, top_n=3)

# KG 인덱스 로드 또는 생성
kg_index, retriever = KGQueryEngine.get_or_create_kg(documents, rerank, save_path="./kg_index.json")
print("show kg: ")
print(kg_index._graph_store._data.graph_dict)
print("show templete: ")
print(kg_index.kg_triplet_extract_template.get_template())

# Chat 엔진 생성
chat_session = KGChatEngine(retriever=retriever, llm=settting.llm, query_wrapper_prompt=query_wrapper_prompt)

# 대화 루프 시작
print("질문을 입력하세요. 0을 입력하면 종료됩니다.")
while True:
    user_input = input("\n질문: ")
    if user_input.lower() in ['0', "exit", "종료", "quit", "그만"]:
        print("대화를 종료합니다.")
        break
    start = time.time()
    response = chat_session.ask(user_input)
    print("답변:", response)
    print("Elapsed:", round(time.time() - start, 2), "s")
