# from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from config import FAST_EMBED_MODEL_NAME
from langchain_huggingface import HuggingFaceEmbeddings

class EmbedSettings:
    def __init__(self):
        self.embed_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            encode_kwargs={"normalize_embeddings": True}
        )
        self.chunk_size = 1024        # 한 청크 크기
        self.chunk_overlap = 256      # 청크 오버랩

    def set_and_get_llama_settings(self):
        print("Setting Llama settings...")
        Settings.embed_model = self.embed_model
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
        return Settings
