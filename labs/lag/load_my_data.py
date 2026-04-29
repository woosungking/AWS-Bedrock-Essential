import os
import boto3
import chromadb
from chromadb.utils.embedding_functions import AmazonBedrockEmbeddingFunction

# 이 스크립트 파일 기준 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------
# 1. 텍스트 파일 읽기
#    - 내가 넣고 싶은 내용을 my_data.txt에 작성
#    - 빈 줄(\n\n) 기준으로 문단 분리
# ----------------------------------------
with open(os.path.join(BASE_DIR, "my_data.txt")) as f:
    content = f.read()

chunks = [c.strip() for c in content.split("\n\n") if c.strip()]

print(f"총 {len(chunks)}개 문단 발견")
for i, chunk in enumerate(chunks):
    print(f"  [{i+1}] {chunk[:50]}...")


# ----------------------------------------
# 2. Bedrock Embedding 함수 준비
#    - 텍스트 → 숫자 벡터로 변환해주는 함수
#    - Titan 모델 사용
# ----------------------------------------
session = boto3.Session()

embedding_function = AmazonBedrockEmbeddingFunction(
    session=session,
    model_name="amazon.titan-embed-text-v2:0"
)


# ----------------------------------------
# 3. ChromaDB 연결
#    - chroma/ 폴더에 로컬 DB 생성
# ----------------------------------------
client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma"))


# ----------------------------------------
# 4. 컬렉션 초기화
#    - 기존 컬렉션 삭제 후 새로 생성
#    - 컬렉션 = DB의 테이블 개념
# ----------------------------------------
try:
    client.delete_collection("my_collection")
    print("\n기존 컬렉션 삭제 완료")
except:
    pass

collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=embedding_function
)
print("새 컬렉션 생성 완료")


# ----------------------------------------
# 5. 데이터 삽입
#    - 각 문단을 컬렉션에 INSERT
#    - embedding은 자동으로 계산됨 (Titan 모델이 처리)
# ----------------------------------------
print("\n데이터 삽입 중...")

for i, chunk in enumerate(chunks):
    collection.add(
        ids=[str(i + 1)],           # 고유 ID
        documents=[chunk],           # 실제 텍스트
        metadatas=[{"source": "my_data.txt"}]  # 부가 정보
    )
    print(f"  [{i+1}] 삽입 완료: {chunk[:40]}...")

print(f"\n최종 저장된 항목 수: {collection.count()}개")

