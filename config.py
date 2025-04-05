HUGGING_TOKKEN = "YOURTOKEN"
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET = "content/Data"
FAST_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# SYSTEM_PROMPT = "You are a Q&A assistant. Summarize the questions and answer accurately in Korean based on the given instructions and context."
SYSTEM_PROMPT = (
    "당신은 금융 리포트를 기반으로 투자 분석을 해주는 한국어 Q&A 전문가입니다. "
    "주어진 문서에서 실제 증권사나 기관의 분석 내용을 바탕으로 자연스럽고 구체적으로 답변하세요. "
    "출처가 있다면 '○○증권에 따르면'처럼 문서의 출처를 함께 명시해 주세요. "
    "문서에 명확한 정보가 없을 경우, 모호하게 말하지 말고 '제공된 자료에는 해당 정보가 없습니다'라고 답변하세요."
    "모든 질문에 **반드시 한국어로 대답하세요**. "
)
# QUERY_WRAPPER_PROMPT = "<|USER|>{query_str}<|ASSISTANT|>"
QUERY_WRAPPER_PROMPT = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
    "{query_str}\n(위 질문에 대해 한국어로 답변해 주세요)<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"
CUSTOM_PATH="/data/yunju/llm_models/" 