class KGChatEngine:
    def __init__(self, retriever, llm, query_wrapper_prompt):
        self.retriever = retriever
        self.llm = llm
        self.query_wrapper_prompt = query_wrapper_prompt

    def ask(self, question: str):
        # 1. triple 기반 context 추출
        retrieved_nodes = self.retriever.retrieve(question)
        context = "\n".join([node.get_content() for node in retrieved_nodes])

        # 2. 문맥 + 질문 포함해서 프롬프트 구성
        full_prompt = self.query_wrapper_prompt.format(
            query_str=f"다음 문서를 참고해서 질문에 답변하세요:\n{context}\n\n질문: {question}"
        )

        # 3. LLM 직접 호출
        response = self.llm.complete(full_prompt)
        return response.text
