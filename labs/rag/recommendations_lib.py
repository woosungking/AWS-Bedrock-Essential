import boto3
import chromadb
from chromadb.utils.embedding_functions import AmazonBedrockEmbeddingFunction

# 1. DB 컬렉션 가져오기
def get_collection(path, collection_name):
    session = boto3.Session()
    embedding_function = AmazonBedrockEmbeddingFunction(session=session, model_name="amazon.titan-embed-text-v2:0")
    
    client = chromadb.PersistentClient(path=path)
    collection = client.get_collection(collection_name, embedding_function=embedding_function)
    
    return collection

# 2. 벡터 검색 실행
def get_vector_search_results(collection, question):
    results = collection.query(
        query_texts=[question],
        n_results=4
    )
    return results

# 3. LLM을 이용한 개별 맞춤형 요약 생성
def get_personalized_recommendation(question, description):
    session = boto3.Session()
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='ap-northeast-2')
    
    message = {
        "role": "user",
        "content": [
            { "text": f"<service_description>{description}</service_description>" },
            { "text": "위 서비스 설명을 바탕으로, 아래 요구사항을 어떻게 해결할 수 있는지 요약해줘. 답변은 반드시 한국어로 작성해줘." },
            { "text": f"<requirements>{question}</requirements>" }
        ]
    }
    
    response = bedrock.converse(
        modelId="apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[message],
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0,
            "topP": 0.9,
        },
    )
    
    return response['output']['message']['content'][0]['text']

# 4. 메인 서비스 로직 (에러 수정 포인트)
def get_similarity_search_results(question):
    # DB 연결
    collection = get_collection("./chroma", "my_collection")
    
    # [Step 1] 검색 실행
    search_results = get_vector_search_results(collection, question)
    
    # 검색된 문서 리스트 추출
    documents = search_results.get('documents', [[]])[0]
    # 메타데이터 리스트 추출 (없으면 빈 리스트로 초기화하여 IndexError 방지)
    metadatas = search_results.get('metadatas', [[]])[0]
    
    num_results = len(documents)
    results_list = []
    
    # [Step 2] 결과만큼 루프 돌며 가공
    for i in range(num_results):
        # LLM 요약 API 호출
        personalized_recommendation = get_personalized_recommendation(question, documents[i])
        
        # --- [에러 해결 구간: 안전하게 데이터 추출] ---
        # i번째 메타데이터 딕셔너리를 가져오되, 없으면 빈 중괄호({}) 리턴
        current_metadata = metadatas[i] if i < len(metadatas) else {}
        
        # .get('key', 'default')를 사용하여 필드가 없어도 터지지 않게 함
        name = current_metadata.get('name', '이름 정보 없음')
        url = current_metadata.get('url', '#')
        # ---------------------------------------------
        
        results_list.append({
            'original': documents[i],
            'summary': personalized_recommendation,
            'name': name,
            'url': url,
        })
    
    return results_list
