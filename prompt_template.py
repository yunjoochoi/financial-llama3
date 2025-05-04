from llama_index.core import PromptTemplate
from config import SYSTEM_PROMPT, QUERY_WRAPPER_PROMPT


class PromptTemplates:
    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT
        self.query_wrapper_prompt = PromptTemplate(QUERY_WRAPPER_PROMPT)
    def get_system_prompt(self):
        print("PromptTemplates.get_system_prompt()")
        return self.system_prompt
    
    def get_query_wrapper_prompt(self):
        print("PromptTemplates.get_query_wrapper_prompt()")
        return self.query_wrapper_prompt