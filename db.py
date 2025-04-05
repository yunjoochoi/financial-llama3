from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

class Db:
    def __init__(self, documents):
        # 메모리에 저장하는 ChromaDB 클라이언트 생성
        chroma_client = chromadb.Client() # 임시 세션용-코드 끝나면 데이터 다 날아감 추후 PersistentClient 등으로 교체

        # ChromaDB 컬렉션 생성
        chroma_collection = chroma_client.create_collection(name="test")

        # ChromaVectorStore 객체 생성
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # StorageContext 설정
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        self.documents = documents
        
    def get_index(self):
        # print("doc 확인: ", self.documents)



        return VectorStoreIndex.from_documents( # self.documents에 있는 문서들을 청킹 + 임베딩해서 디비에 저장하고, 그걸 검색 가능한 인덱스로 만듬 //풀 자동화 api
            self.documents,
            storage_context=self.storage_context
        )
# 기본 청크 배치값: insert_batch_size: int = 2048 / 청크가 만들어진 후, 디비에 저장할 때 한 번에 몇 개를 보낼지 정함
"""
self.documents 예시:
 [Document(id_='6037744d-6bc8-4896-9266-6e0780f3d081', embedding=None, metadata={'file_path': '/home/yunju/text_project/llama3/content/Data/리포트1.txt', 'file_name': '리포트1.txt', 'file_type': 'text/plain', 'file_size': 1048, 'creation_date': '2025-04-05', 'last_modified_date': '2025-04-05'}, excluded_embed_metadata_keys=['file_name', 'file_type', 'file_size', 'creation_date', 'last_modified_date', 'last_accessed_date'], excluded_llm_metadata_keys=['file_name', 'file_type', 'file_size', 'creation_date', 'last_modified_date', 'last_accessed_date'], relationships={}, metadata_template='{key}: {value}', metadata_separator='\n', text_resource=MediaResource(embeddings=None, data=None, text='DS투자증권\tabout\tSK하이닉스\r\nDS투자증권\taboutIndustry\t반도체\r\nDS투자증권\tstatesTargetPrice\t290000\r\nDS투자증권\tstatesCurrentPrice\t225500\r\nDS투자증권\tstatesOpinion\t매수\r\nDS투자증권\tmentionsPositiveReason\tHBM 매출 기여로 이익 가시성 개선\r\nDS투자증권\tmentionsPositiveReason\t2025년 HBM 매출 YoY +100% 이상 성장 전망\r\nDS투자증권\tmentionsPositiveReason\tAI 시장 성장과 커스텀 HBM 개화\r\nDS투자증권\tmentionsPositiveReason\t2026년 HBM 공급계획 상반기 결정 예정\r\nDS투자증권\tmentionsRiskFactor\tNAND 부문 부진 및 회복 지연\r\nDS투자증권\tstatesForecastRevenue2024\t66193십억원\r\nDS투자증권\tstatesForecastOP2024\t23468십억원\r\nDS투자증권\tstatesEPS2024\t23362\r\nDS투자증권\tstatesPER2024\t9.7배\r\nDS투자증권\tstatesROE2024\t27.60%\r\nDS투자증권\tstatesForecastRevenue2025\t85184십억원\r\nDS투자증권\tstatesForecastOP2025\t33818십억원\r\nDS투자증권\tstatesEPS2025\t33369\r\nDS투자증권\tstatesPER2025\t6.8배\r\nDS투자증권\tstatesROE2025\t29.90%\r\n', path=None, url=None, mimetype=None), image_resource=None, audio_resource=None, video_resource=None, text_template='{metadata_str}\n\n{content}')]
=> 리스트 요소 하나하나가 BaseNode(벡터 DB에 저장되는 기본 단위), 
=> embedding필드가 text를 임베딩한 벡터로, 검색시 사용됨. 
"""

"""
잘 작동 X - 각Extractor에 작동하는 프롬프트 수정해야할듯
from llama_index.core.extractors import (
            SummaryExtractor,
            QuestionsAnsweredExtractor,
            TitleExtractor,
            KeywordExtractor,
            # EntityExtractor,
        )
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core import VectorStoreIndex

        from llama_index.core.node_parser import SentenceSplitter

        transformations = [
            SentenceSplitter(),  # 먼저 문장을 TextNode로 나누기
            TitleExtractor(nodes=5),
            QuestionsAnsweredExtractor(questions=3),
            SummaryExtractor(summaries=["prev", "self"]),
            KeywordExtractor(keywords=10),
            # EntityExtractor 제거했음
        ]

        pipeline = IngestionPipeline(transformations=transformations)
        nodes = pipeline.run(documents=self.documents)
        print("nodes확인: ", nodes)

        return VectorStoreIndex(nodes=nodes)
"""