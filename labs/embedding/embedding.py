import json
import boto3
from numpy import dot
from numpy.linalg import norm

# -------------------------------
# 텍스트 + 임베딩을 함께 저장하는 클래스
# 생성 시 바로 embedding을 생성함 (API 호출 발생)
# -------------------------------
class EmbedItem:
    def __init__(self, text):
        self.text = text                     # 원본 텍스트
        self.embedding = get_embedding(text)  # 텍스트 → 벡터 변환

# -------------------------------
# 유사도 계산 결과 저장용 클래스
# -------------------------------
class ComparisonResult:
    def __init__(self, text, similarity):
        self.text = text            # 비교 대상 텍스트
        self.similarity = similarity  # 유사도 점수

# -------------------------------
# 코사인 유사도 계산 함수
# 두 벡터가 얼마나 비슷한 방향인지 계산
# -------------------------------
def calculate_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# -------------------------------
# Bedrock 임베딩 호출 함수
# 텍스트를 벡터(숫자 배열)로 변환
# -------------------------------
def get_embedding(text):
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime')
    
    response = bedrock.invoke_model(
        body=json.dumps({ "inputText": text }),  # 입력 텍스트
        modelId="amazon.titan-embed-text-v2:0",  # 임베딩 모델
        accept="application/json",
        contentType="application/json"
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['embedding']  # 벡터 반환

# -------------------------------
# 1. 비교할 텍스트 목록 읽기
# -------------------------------
items = []

with open("items.txt", "r") as f:
    text_items = f.read().splitlines()  # 한 줄씩 리스트로 읽음

# -------------------------------
# 2. 각 텍스트를 embedding으로 변환
# (여기서 텍스트 개수만큼 Bedrock API 호출 발생)
# -------------------------------
for text in text_items:
    items.append(EmbedItem(text))

# -------------------------------
# 3. 모든 텍스트 쌍을 비교
# -------------------------------
for e1 in items:
    print(f"Closest matches for '{e1.text}'")
    print("----------------")
    
    cosine_comparisons = []  # 비교 결과 저장 리스트
    
    for e2 in items:
        # 두 텍스트 간 유사도 계산
        similarity_score = calculate_similarity(e1.embedding, e2.embedding)
        
        # 결과 저장
        cosine_comparisons.append(
            ComparisonResult(e2.text, similarity_score)
        )
    
    # -------------------------------
    # 4. 유사도 높은 순으로 정렬
    # -------------------------------
    cosine_comparisons.sort(
        key=lambda x: x.similarity,
        reverse=True
    )
    
    # -------------------------------
    # 5. 결과 출력
    # -------------------------------
    for c in cosine_comparisons:
        print("%.6f" % c.similarity, "\t", c.text)
    
    print()
