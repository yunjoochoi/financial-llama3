# ✅ 개선된 KG_graph.py - dict(json) 기반 KG 복원 지원
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import TextNode
import os
import json

# 사용자 정의 템플릿
TRIPLE_EXTRACTION_TEMPLATE = """
당신은 금융 리포트에서 고품질 지식 그래프를 추출하는 전문가입니다.

다음 문단을 읽고, 정확하고 구체적인 (주어, 관계, 객체) 삼중항을 최대 3개까지 추출하세요.

조건:
- 관계는 '추정', '적용', '관계' 같은 추상 표현 대신 구체적으로 표현 (ex: '영업이익 증가', '목표주가 하향')
- 숫자/기호만 있는 항목은 제외
- 날짜/수치는 목적어에만 허용
- 명확한 주어가 없는 정보는 생략
- 불확실하거나 예측 불가한 내용은 생략
- 관계는 가능하면 동사 + 명사 형태로 기술 (예: '실적 발표', '투자 의견 변경')

예시:
문장: "SK하이닉스는 2025년 영업이익이 10조 원을 넘을 것이라고 전망했다."
→ ('SK하이닉스', '2025년 영업이익 전망', '10조 원')

문장:
{text}
"""


# JSON 기반 저장
def save_kg(triples_dict, path="kg_index.json"):
    print(f"KG triple을 dict(json) 형식으로 저장합니다: {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(triples_dict, f, ensure_ascii=False, indent=2)

def load_kg(path="kg_index.json"):
    print(f"저장된 KG 로딩 중: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class KGQueryEngine:
    @staticmethod
    def get_or_create_kg(documents, rerank, save_path="kg_index.json"):
        if os.path.exists(save_path):
            graph_dict = load_kg(save_path)
            triples = []
            for subj, preds in graph_dict.items():
                for pred, obj in preds:
                    triples.append(f"{subj}\t{pred}\t{obj}")
            node = TextNode(text="\n".join(triples))
            kg_index = KnowledgeGraphIndex.from_documents([node])
        else:
            print("Knowledge Graph 생성 중...")
            kg_index = KnowledgeGraphIndex.from_documents(
                documents,
                max_triplets_per_chunk=3,
                include_embeddings=False,
                kg_triplet_extract_template=PromptTemplate(TRIPLE_EXTRACTION_TEMPLATE)
            )
            save_kg(kg_index._graph_store._data.graph_dict, save_path)

        retriever = kg_index.as_retriever(
            retriever_mode="keyword",
            similarity_top_k=10,
            node_postprocessors=[rerank] if rerank else []
        )
        return kg_index, retriever
    def print_triples(self):
        print("Triple: ")
        print(self.kg_index._graph_store._data.graph_dict)

    def show_prompt(self):
        print("Triple Extraction Prompt:")
        print(self.kg_index.kg_triplet_extract_template.get_template())