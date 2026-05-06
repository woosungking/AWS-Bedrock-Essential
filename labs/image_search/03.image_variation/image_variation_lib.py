import boto3
import json
import base64
from io import BytesIO
from random import randint

def get_bytesio_from_bytes(image_bytes):
    """
    파일 바이트 데이터로부터 BytesIO 객체를 반환합니다.
    
    Args:
        image_bytes (bytes): 로컬 파일 등에서 읽어들인 원본 이미지의 이진 데이터 (0과 1의 덩어리)
    """
    image_io = BytesIO(image_bytes)
    return image_io

def get_base64_from_bytes(image_bytes):
    """
    파일 바이트 데이터로부터 Base64 인코딩된 문자열을 반환합니다.
    
    Args:
        image_bytes (bytes): Base64 텍스트로 변환할 원본 이미지의 이진 데이터
    """
    resized_io = get_bytesio_from_bytes(image_bytes)
    img_str = base64.b64encode(resized_io.getvalue()).decode("utf-8")
    return img_str

def get_bytes_from_file(file_path):
    """
    디스크의 파일에서 바이트 데이터를 읽어옵니다.
    
    Args:
        file_path (str): 읽어올 이미지 파일이 저장된 로컬 PC의 경로 (예: "my_picture.jpg" 또는 "./images/test.png")
    """
    with open(file_path, "rb") as image_file:
        file_bytes = image_file.read()
    return file_bytes

def get_image_variation_request_body(prompt, similarity_strength, image_bytes=None):
    """
    InvokeModel API 호출에 사용할 JSON 요청 본문을 생성합니다. (Nova Canvas 공식 규격 적용)
    
    Args:
        prompt (str): 원본 이미지를 바탕으로 새롭게 추가/변경하고 싶은 내용을 설명하는 명령어 (예: "배경을 겨울로 바꿔줘")
        similarity_strength (float): 원본 이미지를 얼마나 유지할지 결정하는 수치 (0.2 ~ 1.0 사이 권장. 높을수록 원본과 거의 동일하게 생성됨)
        image_bytes (bytes, optional): 변형할 원본 이미지의 이진 데이터. (Bedrock 서버 전송을 위해 내부에서 Base64로 변환됨)
    """
    input_image_base64 = get_base64_from_bytes(image_bytes)
    
    body = { 
        "taskType": "IMAGE_VARIATION",  # 에러 해결: 카멜케이스 적용
        "imageVariationParams": {       # 에러 해결: 공식 파라미터명 적용
            "images": [
                input_image_base64
            ], 
            "text": prompt,  
            "similarityStrength": similarity_strength 
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,  
            "quality": "standard",  
            "width": 512,
            "height": 512,
            "cfgScale": 8.0, 
            "seed": randint(0, 100000),  
        },
    }
    
    return json.dumps(body)

def get_response_image(response):
    """
    Nova Canvas 응답을 파싱하여 생성된 이미지를 추출합니다.
    
    Args:
        response (dict): boto3의 `invoke_model` 함수가 반환한 Bedrock API의 원본 응답 객체
    """
    response = json.loads(response.get('body').read())
    
    images = response.get('images')
    
    image_data = base64.b64decode(images[0])

    return BytesIO(image_data)

def get_image_from_model(prompt_content, similarity_strength, image_bytes):
    """
    Amazon Nova Canvas를 사용하여 이미지 변형(Variation)을 생성하는 메인 실행 함수입니다.
    
    Args:
        prompt_content (str): 이미지에 적용할 변경 사항을 설명하는 텍스트 프롬프트
        similarity_strength (float): 원본 이미지와의 유사도 유지 강도 (0.0 ~ 1.0)
        image_bytes (bytes): 변형의 기준이 될 원본 이미지 데이터 (get_bytes_from_file 함수로 읽어온 값)
    """
    session = boto3.Session()

    # Nova Canvas 모델을 안정적으로 지원하는 버지니아 북부(us-east-1) 리전 사용
    bedrock = session.client(service_name='bedrock-runtime', region_name='us-east-1') 
    
    body = get_image_variation_request_body(prompt_content, similarity_strength, image_bytes)
    
    response = bedrock.invoke_model(
        body=body, 
        modelId="amazon.nova-canvas-v1:0", 
        contentType="application/json", 
        accept="application/json"
    )
    
    output = get_response_image(response)
    
    return output