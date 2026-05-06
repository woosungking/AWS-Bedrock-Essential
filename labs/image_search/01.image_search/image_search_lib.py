import itertools
import boto3
import json
import base64
import chromadb
from io import BytesIO
from PIL import Image
# 1. 컬렉션 로드 함수 (상대 경로로 수정 권장)
def get_collection(path, collection_name):
    # 경로를 실제 DB가 저장된 위치인 "./chroma_db" 등으로 확인하세요.
    client = chromadb.PersistentClient(path=path)
    collection = client.get_collection(collection_name)
    return collection

# 2. 벡터 검색 함수 (들여쓰기 수정 완료)
def get_vector_search_results(collection, query_embedding):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4
    )
    return results

# 3. 이미지 바이트를 Base64 문자열로 변환
def get_base64_from_bytes(image_bytes):
    image_io = BytesIO(image_bytes)
    image_base64 = base64.b64encode(image_io.getvalue()).decode("utf-8")
    return image_base64

# 4. Bedrock을 호출하여 멀티모달 벡터 생성
def get_multimodal_vector(input_image_base64=None, input_text=None):
    # 리전 권한에 따라 region_name='us-east-1' 추가 필요할 수 있음
    session = boto3.Session(region_name='us-east-1') 
    bedrock = session.client(service_name='bedrock-runtime')
    
    request_body = {}
    if input_text:
        request_body["inputText"] = input_text
    if input_image_base64:
        request_body["inputImage"] = input_image_base64
    
    body = json.dumps(request_body)
    
    response = bedrock.invoke_model(
        body=body, 
        modelId="amazon.titan-embed-image-v1", 
        accept="application/json", 
        contentType="application/json"
    )
    
    response_body = json.loads(response.get('body').read())
    return response_body.get("embedding")

# 5. 핵심 검색 로직 (진입점 역할)
def get_similarity_search_results(search_term=None, search_image=None):
    # 이미지가 들어왔다면 Base64로 변환
    search_image_base64 = (get_base64_from_bytes(search_image) if search_image else None)

    # 텍스트/이미지로 쿼리 벡터 생성
    query_embedding = get_multimodal_vector(input_text=search_term, input_image_base64=search_image_base64)
    
    # DB 연결 (아까 만든 chroma_db 폴더 경로와 이름을 맞춰주세요)
    # 예: "./chroma_db", "image_collection"
    collection = get_collection("./chroma_db", "image_collection")
    
    # 유사도 검색 수행
    search_results = get_vector_search_results(collection, query_embedding)
    
    # 결과 리스트 평탄화 (ChromaDB 결과는 리스트의 리스트 형태임)
    flattened_results_list = list(itertools.chain(*search_results['documents']))
    
    results_images = []
    for res in flattened_results_list:
        try:
            with open(res, "rb") as f: 
                img = BytesIO(f.read())
            results_images.append(img)
        except FileNotFoundError:
            print(f"이미지 파일을 찾을 수 없습니다: {res}")
    
    return results_images

if __name__ == "__main__":
    search_keyword = "a photo of a cat"
    found_images = get_similarity_search_results(search_term=search_keyword)
    
    for i, img_data in enumerate(found_images):
        with open(f"search_result_{i}.jpg", "wb") as f:
            f.write(img_data.getbuffer()) # BytesIO에서 데이터를 꺼내 저장
    
    print(f"{len(found_images)}개의 이미지를 저장했습니다. 폴더를 확인하세요!")