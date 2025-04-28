from llama_index.core import Settings
import torch
from typing import List

class KeywordExtractor:
    def __init__(self, top_k=5):
        # BaseEmbedding 타입 지정 없이 그냥 embed_model 사용
        self.embed_model = Settings.embed_model
        self.top_k = top_k

    def extract(self, question: str) -> List[str]:
        candidates = question.split()
        question_emb = self.embed_model.get_text_embedding(question)
        candidate_embs = [self.embed_model.get_text_embedding(c) for c in candidates]
        sims = [self.cosine_similarity(question_emb, cand_emb) for cand_emb in candidate_embs]
        sorted_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        top_keywords = [candidates[i] for i in sorted_indices[:self.top_k]]
        return top_keywords

    def cosine_similarity(self, emb1, emb2):
        emb1 = torch.tensor(emb1)
        emb2 = torch.tensor(emb2)
        return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()

# # 사용 예시
# extractor = KeywordExtractor(top_k=3)
# query = "하이닉스의 2025년 영업이익 전망은 어떻게 될까?"
# keywords = extractor.extract(query)

# print("추출된 키워드:", keywords)
