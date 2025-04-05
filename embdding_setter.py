# from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from config import FAST_EMBED_MODEL_NAME
from langchain_huggingface import HuggingFaceEmbeddings

class EmbedSettings:
  def __init__(self):
    # self.embed_model = FastEmbedEmbedding(model_name=FAST_EMBED_MODEL_NAME)
    self.embed_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small",
                                         encode_kwargs={"normalize_embeddings": True}) # 성능 향상 옵션)
    self.chunk_size = 512 # 임베딩 전 토큰 몇개 쓸건지 (# of input)
 
  def set_and_get_llama_settings(self):
    print("Setting Llama settings...")
    Settings.embed_model = self.embed_model
    Settings.chunk_size = self.chunk_size
    return Settings