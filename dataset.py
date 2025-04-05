from llama_index.core import SimpleDirectoryReader
from config import DATASET
class DataSetter:
  def __init__(self):
    self.dataset_dir = DATASET

  def load_dataset(self):
    print("Loading dataset...")
    return SimpleDirectoryReader(self.dataset_dir).load_data()
  
  """
  SimpleDirectoryReader?
  지정한 디렉토리 내의 .txt, .pdf, .md, .json 등의 파일을 읽어서,
  LlamaIndex에서 사용할 수 있는 Document 객체로 변환
  """