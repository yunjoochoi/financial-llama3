from huggingface_hub import login
from config import HUGGING_TOKKEN

class HuggingFaceNotebookLogin:
    def __init__(self ):
        self.token = HUGGING_TOKKEN
    def login(self):
        print("Logging in...")
        login(self.token)