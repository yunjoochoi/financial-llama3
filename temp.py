from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex

import time
from hugging import HuggingFaceNotebookLogin
from dataset  import DataSetter
from embdding_setter import EmbedSettings
from load_llm import LLMLoader
from prompt_template import PromptTemplates
from db import Db
from config import ENCODER_MODEL

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.font_manager as fm

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

print("Setting KG graph...")
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=5,
    include_embeddings=True  # hybrid 검색도 가능 - 임베딩 기반 + triple 기반
)
# 한글 폰트 경로 설정
# font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # 리눅스 기준
# font_prop = fm.FontProperties(fname=font_path, size=10)
# plt.rcParams["font.family"] = font_prop.get_name()
# G = kg_index.get_networkx_graph()
# plt.figure(figsize=(10, 8))
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, edge_color="gray")
# plt.title("Knowledge Graph")
# plt.savefig("knowledge_graph.png")
# print("knowledge_graph.png 저장 완료")

print("추출된 kg 구조:")
print(kg_index._graph_store._data.graph_dict)

print("Triple 추출하는데 사용된 Prompt 확인")
print(kg_index.kg_triplet_extract_template.get_template())

print("Setting Query Engine settings...")
query_engine = kg_index.as_query_engine(similarity_top_k=5)


print("Query Engine Start...")
now = time.time()
response = query_engine.query("하이닉스 영업이익 추정치는?")
print(f"Response Generated: {response}")
print(f"Elapsed: {round(time.time() - now, 2)}s")