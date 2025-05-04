import os
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
HUGGING_TOKEN = os.getenv("HUGGING_TOKEN")
login(token=HUGGING_TOKEN)

class HuggingFaceNotebookLogin:
    def __init__(self ):
        self.token = HUGGING_TOKEN
    def login(self):
        print("Logging in...")
        login(self.token)


