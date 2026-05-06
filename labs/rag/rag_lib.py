import itertools
import json
import boto3
import chromadb
from chromadb.utils.embedding_functions import AmazonBedrockEmbeddingFunction

# rag_lib.py 기준 상대경로
import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_FILE = os.path.join(_BASE_DIR, "bedrock_faqs_with_embeddings.json")


# -------------------------------
# ChromaDB에서 컬렉션 가져오기
# 없으면 자동으로 생성 후 데이터 로드
# -------------------------------
def get_collection(path, collection_name):
    session = boto3.Session()

    # 텍스트 → embedding 자동 생성 함수 (Bedrock 사용)
    embedding_function = AmazonBedrockEmbeddingFunction(
        session=session,
        model_name="amazon.titan-embed-text-v2:0"
    )
    
    # 디스크 기반 ChromaDB 연결
    client = chromadb.PersistentClient(path=path)

    # 컬렉션이 없으면 자동 생성 후 데이터 로드
    collection = client.get_or_create_collection(
        collection_name,
        embedding_function=embedding_function
    )

    if collection.count() == 0:
        print(f"[초기화] {collection_name} 컬렉션이 비어있어 데이터를 로드합니다...")
        with open(_DATA_FILE) as f:
            source_json = json.load(f)
        for item in source_json:
            collection.add(
                ids=[str(item['id'])],
                documents=[item['document']],
                metadatas=[item['metadata']],
                embeddings=[item['embedding']]
            )
        print(f"[완료] {collection.count()}개 항목 로드됨")
    
    return collection


# -------------------------------
# 벡터 검색 수행
# 질문과 유사한 문서 top-k 반환
# -------------------------------
def get_vector_search_results(collection, question):
    
    results = collection.query(
        query_texts=[question],  # 질문을 embedding으로 변환 후 검색
        n_results=4              # 가장 유사한 4개 반환
    )
    
    return results


# -------------------------------
# RAG 전체 실행 함수
# -------------------------------
def get_rag_response(question):

    # Bedrock LLM 호출용 클라이언트
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime')
    
    # -------------------------------
    # 1. Vector DB에서 컬렉션 가져오기
    # -------------------------------
    collection = get_collection(
        "chroma",
        "my_collection"
    )
    
    # -------------------------------
    # 2. 질문과 유사한 문서 검색
    # -------------------------------
    search_results = get_vector_search_results(collection, question)
    
    # -------------------------------
    # 3. 결과 구조 flatten
    # (Chroma는 [[doc1, doc2, ...]] 형태로 반환)
    # -------------------------------
    flattened_results_list = list(
        itertools.chain(*search_results['documents'])
    )
    
    # -------------------------------
    # 4. 검색된 문서들을 하나의 텍스트로 합침
    # (RAG context 생성)
    # -------------------------------
    rag_content = "\n\n".join(flattened_results_list)

    # 디버깅용 출력 (LLM에 들어갈 context 확인)
    print(rag_content)
    
    # -------------------------------
    # 5. LLM에 전달할 메시지 구성
    # -------------------------------
    message = {
        "role": "user",
        "content": [
            { "text": rag_content },  # 🔥 검색된 문서 (근거 데이터)
            { "text": "Based on the content above, please answer the following question:" },
            { "text": question }      # 실제 질문
        ]
    }
    
    # -------------------------------
    # 6. Bedrock LLM 호출
    # -------------------------------
    response = bedrock.converse(
        modelId="apac.amazon.nova-lite-v1:0",  # ap-northeast-2 (서울) cross-region inference
        messages=[message],
        inferenceConfig={
            "maxTokens": 2000,   # 최대 응답 길이
            "temperature": 0,    # deterministic (정확성 위주)
            "topP": 0.9,
            "stopSequences": []
        },
    )
    
    # -------------------------------
    # 7. 결과 반환
    # -------------------------------
    return (
        response['output']['message']['content'][0]['text'],  # LLM 답변
        flattened_results_list                                # 근거 문서
    )
