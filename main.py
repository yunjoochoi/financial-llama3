from llama_index.core.postprocessor import SentenceTransformerRerank


import time
from hugging import HuggingFaceNotebookLogin
from dataset  import DataSetter
from embdding_setter import EmbedSettings
from load_llm import LLMLoader
from prompt_template import PromptTemplates
from db import Db
from config import ENCODER_MODEL


print("hugging Login Start...")
HuggingFaceNotebookLogin().login()

print("load PDF Data Start...")
documents = DataSetter().load_dataset()

promptTP = PromptTemplates()
system_prompt = promptTP.get_system_prompt()
query_wrapper_prompt = promptTP.get_query_wrapper_prompt()

print("Setting Embed Model settings...")
settting = EmbedSettings().set_and_get_llama_settings()

print("Setting LLM settings...")
settting.llm = LLMLoader(system_prompt, query_wrapper_prompt).load_llm()

# 쿼리에서 임베딩 기반 키워드 추출
from keyword_extractor import KeywordExtractor
extractor = KeywordExtractor(top_k=3)
query = "하이닉스의 2025년 영업이익 전망은 어떻게 될까?"
keywords = extractor.extract(query)

print("추출된 키워드:", keywords)
print("Setting db Index settings...")
dbIndex = Db(documents).get_index()

print("Setting SentenceTransformerRerank settings...")
rerank = SentenceTransformerRerank(model=ENCODER_MODEL, top_n=3)

print("Setting Query Engine settings...")
query_engine = dbIndex.as_query_engine(similarity_top_k=10, node_postprocessors=[rerank])

print("Query Engine Start...")
now = time.time()
response = query_engine.query("2025년도 하이닉스 영업이익 추정치는?")
print(f"Response Generated: {response}")
print(f"Elapsed: {round(time.time() - now, 2)}s")