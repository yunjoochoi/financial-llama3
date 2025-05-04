from llama_index.core.postprocessor import SentenceTransformerRerank
import time
from hugging import HuggingFaceNotebookLogin
from dataset import DataSetter
from embdding_setter import EmbedSettings
from load_llm import LLMLoader
from prompt_template import PromptTemplates
from db import Db
from config import ENCODER_MODEL

from sentence_transformers import SentenceTransformer, util
import torch

# smart reranking 함수 (코사인 유사도 + 중복 가산점)
def smart_rerank_context(original_response, keyword_response, question, top_k=5):
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    query_emb = model.encode(question, convert_to_tensor=True)

    combined_nodes = (original_response.source_nodes if hasattr(original_response, "source_nodes") else []) + \
                     (keyword_response.source_nodes if hasattr(keyword_response, "source_nodes") else [])

    node_texts = list({node.get_text().strip(): node for node in combined_nodes}.keys())
    node_embs = model.encode(node_texts, convert_to_tensor=True)

    scores = util.cos_sim(query_emb, node_embs)[0]

    node_score_map = {}
    for idx, text in enumerate(node_texts):
        base_score = scores[idx].item()
        bonus = 0
        if any(text == node.get_text().strip() for node in original_response.source_nodes) and \
           any(text == node.get_text().strip() for node in keyword_response.source_nodes):
            bonus += 0.2  # ✅ 여기서 중복 가산점 0.2로 세팅 (조정 가능)

        node_score_map[text] = base_score + bonus

    sorted_texts = sorted(node_score_map.items(), key=lambda x: x[1], reverse=True)
    final_texts = [text for text, _ in sorted_texts[:top_k]]

    all_nodes_map = {node.get_text().strip(): node for node in combined_nodes}
    final_nodes = [all_nodes_map[text] for text in final_texts if text in all_nodes_map]

    print(f"✅ 최종 선택된 context 수: {len(final_nodes)}개")
    return final_nodes


### ----- Main 실행 -----

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

from keyword_extractor import KeywordExtractor
extractor = KeywordExtractor(top_k=6)

print("Setting db Index settings...")
dbIndex = Db(documents).get_index()

print("Setting SentenceTransformerRerank settings...")
rerank = SentenceTransformerRerank(model=ENCODER_MODEL, top_n=5)

print("Setting Query Engine settings...")
query_engine = dbIndex.as_query_engine(similarity_top_k=20, node_postprocessors=[rerank])

# 1. 사용자 질문
user_query = "하이닉스 2024 NAND 세일즈는?"

# 2. 키워드 추출
keywords = extractor.extract(user_query)
print("\n✅ 추출된 키워드:", keywords)

# 3. 키워드 기반 쿼리 생성
keywords_query = " ".join(keywords)
print("✅ 키워드 기반 쿼리:", keywords_query)

# 4. 각각 별도로 검색
print("\nOriginal Query로 검색 중...")
original_response = query_engine.query(user_query)

print("Keywords Query로 검색 중...")
keyword_response = query_engine.query(keywords_query)

# 5. smart rerank 적용해서 최종 context 선택
final_context_nodes = smart_rerank_context(original_response, keyword_response, user_query, top_k=5
                                           )

# 6. 최적화된 context로 LLM에 최종 프롬프트 생성
final_context_text = "\n\n".join([node.get_text() for node in final_context_nodes])
print("\n✅ 최종 답변 생성에 사용된 Context:")
print(final_context_text)

final_prompt = f"""
다음 문서를 참고하여 질문에 정확하고 구체적으로 답변해 주세요:

{final_context_text}

질문: {user_query}
답변:
"""

# 7. LLM을 직접 호출해서 최종 답변 생성
response = settting.llm.complete(final_prompt)

# 8. 최종 결과 출력
print("\n✅ 최종 최적화된 답변:")
print(response.text)

# 9. 참고로 Original / Keyword 별 검색 결과 출력
print("\n📄 Original Query Search 결과:")
if hasattr(original_response, "source_nodes"):
    for idx, node in enumerate(original_response.source_nodes):
        print(f"[Original {idx+1}]")
else:
    print("No original source nodes found.")

print("\n📄 Keyword Query Search 결과:")
if hasattr(keyword_response, "source_nodes"):
    for idx, node in enumerate(keyword_response.source_nodes):
        print(f"[Keyword {idx+1}]\n")
else:
    print("No keyword source nodes found.")
