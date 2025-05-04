from typing import List
from mecab import MeCab
import torch
import re
from llama_index.core import Settings

class KeywordExtractor:
    def __init__(self, top_k=5):
        self.embed_model = Settings.embed_model
        self.top_k = top_k
        self.mecab = MeCab()

    def extract(self, question: str) -> List[str]:
        pos_tagged = self.mecab.pos(question)

        # 복합명사(연속된 명사) + 대문자 알파벳 단어 보존
        candidates = []
        buffer = []

        for word, pos in pos_tagged:
            if pos.startswith('NN') or self.is_uppercase_word(word):
                buffer.append(word)
            else:
                if buffer:
                    candidates.append(''.join(buffer))
                    buffer = []

        if buffer:
            candidates.append(''.join(buffer))

        # 후보가 하나도 없으면 fallback: 띄어쓰기 split
        if not candidates:
            candidates = question.split()

        # Embedding 기반 유사도 점수 계산
        question_emb = self.embed_model.get_text_embedding(question)
        candidate_embs = [self.embed_model.get_text_embedding(c) for c in candidates]

        sims = [self.cosine_similarity(question_emb, cand_emb) for cand_emb in candidate_embs]

        # 상위 top_k 키워드 선택
        sorted_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        top_keywords = [candidates[i] for i in sorted_indices[:self.top_k]]
        return top_keywords

    def is_uppercase_word(self, word: str) -> bool:
        return bool(re.fullmatch(r'[A-Z]+', word))

    def cosine_similarity(self, emb1, emb2):
        emb1 = torch.tensor(emb1)
        emb2 = torch.tensor(emb2)
        return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()


# # 사용 예시
# extractor = KeywordExtractor(top_k=3)
# query = "하이닉스의 2025년 영업이익 전망은 어떻게 될까?"
# keywords = extractor.extract(query)

# print("추출된 키워드:", keywords)
