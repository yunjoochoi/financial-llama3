# ragas 포맷 기반으로 RAG 시스템 평가하는 스크립트
from typing import List, Dict
import pandas as pd
from ragas.metrics import context_recall, faithfulness, answer_relevancy, f1, exact_match
from ragas.evaluation import evaluate
from datasets import Dataset

# ragas 포맷 변환 함수
def load_ragas_data_from_csv(filepath: str) -> List[Dict]:
    """
    CSV 파일을 읽어 RAGAS 포맷에 맞게 변환
    columns: question, contexts, prediction(answer), ground_truth
    contexts 컬럼은 리스트 형태로 저장
    """
    df = pd.read_csv(filepath)
    data = []
    for _, row in df.iterrows():
        contexts = eval(row["contexts"]) if isinstance(row["contexts"], str) else []
        data.append({
            "question": row["question"],
            "contexts": contexts,
            "answer": row["prediction"],
            "ground_truth": row["ground_truth"]
        })
    return data

if __name__ == "__main__":
    filepath = "ragas_predictions.csv"  # <-- ragas 포맷용 CSV 파일명
    ragas_data = load_ragas_data_from_csv(filepath)

    dataset = Dataset.from_list(ragas_data)

    result = evaluate(
        dataset,
        metrics=[context_recall, faithfulness, answer_relevancy, f1, exact_match]
    )

    print(result)
