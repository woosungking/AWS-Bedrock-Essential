import boto3
import json
import base64
import os
import chromadb

session = boto3.Session(region_name='us-east-1') 
bedrock = session.client(service_name='bedrock-runtime')

def get_multimodal_vector(input_image_base64=None, input_text=None):
    request_body = {}
    if input_text: request_body["inputText"] = input_text
    if input_image_base64: request_body["inputImage"] = input_image_base64

    try:
        response = bedrock.invoke_model(
            body=json.dumps(request_body),
            modelId="amazon.titan-embed-image-v1", # 모델 ID가 정확한지 확인
            accept="application/json",
            contentType="application/json"
        )
        return json.loads(response.get('body').read()).get("embedding")
    except Exception as e:
        print(f"모델 호출 중 상세 에러: {e}")
        raise e

def process_and_store_images():
    image_path = "images"
    db_path = "./chroma_db"
    collection_name = "image_collection"

    # ChromaDB 설정
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    if not os.path.exists(image_path):
        print(f"오류: '{image_path}' 폴더가 없습니다.")
        return

    print(f"작업 시작...")

    # 이미 처리된 파일 건너뛰기 로직 (선택 사항)
    existing_count = collection.count()
    
    row_count = 0
    for file in sorted(os.listdir(image_path)): # 순서대로 처리
        if file.startswith('.') or not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        file_path = os.path.join(image_path, file)
        
        try:
            with open(file_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode('utf8')
            
            embedding = get_multimodal_vector(input_image_base64=img_base64)
            
            row_count += 1
            collection.add(
                ids=[f"img_{row_count + existing_count}"], # ID 중복 방지
                documents=[f"images/{file}"],
                metadatas=[{"file_path": f"images/{file}", "file_name": file}],
                embeddings=[embedding]
            )
            print(f"성공 [{row_count + existing_count}]: {file}")

        except Exception as e:
            # 여기서 에러가 나면 모델 ID나 권한 문제일 확률 99%
            print(f"실패: {file} 처리 중 에러 -> {e}")
            break # 치명적 에러 시 루프 중단

    print(f"\n✅ 최종 완료! 현재 DB 내 총 데이터 수: {collection.count()}")

if __name__ == "__main__":
    process_and_store_images()