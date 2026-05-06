# 필수 라이브러리 임포트
import boto3
import json
import base64
from io import BytesIO
from random import randint

# ------------------------------------------------------------------------
# [유틸리티 함수] 이미지/바이트 데이터 변환
# ------------------------------------------------------------------------

def get_bytesio_from_bytes(image_bytes):
    """파일 바이트 데이터를 메모리에서 다룰 수 있는 BytesIO 가상 파일 객체로 변환합니다."""
    image_io = BytesIO(image_bytes)
    return image_io

def get_base64_from_bytes(image_bytes):
    """바이트 데이터를 AWS AI 모델 전송 규격인 Base64 문자열로 인코딩합니다."""
    resized_io = get_bytesio_from_bytes(image_bytes)
    img_str = base64.b64encode(resized_io.getvalue()).decode('utf-8')
    return img_str

def get_bytes_from_file(file_path):
    """로컬 PC의 이미지 파일을 읽어 이진(Bytes) 데이터로 반환합니다."""
    with open(file_path, "rb") as image_file:
        file_bytes = image_file.read()
    return file_bytes

# ------------------------------------------------------------------------
# [메인 로직] API 요청 본문 생성 및 호출
# ------------------------------------------------------------------------

def get_image_background_replacement_request_body(prompt, image_bytes, mask_prompt, negative_prompt=None, outpainting_mode="DEFAULT"):
    """
    Nova Canvas 모델에 보낼 '배경 교체(아웃페인팅)' 전용 JSON 요청 바디를 생성합니다.
    
    Args:
        prompt (str): 새롭게 생성할 배경에 대한 묘사 (예: "눈 덮인 산", "세련된 카페 테이블")
        image_bytes (bytes): 배경을 바꿀 원본 이미지 데이터
        mask_prompt (str): 사진에서 배경을 날려버릴 때 '절대 건드리지 않고 보호할 대상' (예: "커피잔", "강아지")
        negative_prompt (str, optional): 배경에 절대 나오지 않았으면 하는 요소
        outpainting_mode (str): 마스크(보호 영역)의 경계선 처리 방식 ("DEFAULT" 또는 "PRECISE")
    """
    # 1. 원본 이미지를 Base64 문자열로 변환
    input_image_base64 = get_base64_from_bytes(image_bytes)

    # 2. 배경 교체(아웃페인팅) 기본 JSON 뼈대 생성
    body = { 
        "taskType": "OUTPAINTING", # 배경을 바꾸거나 넓힐 때는 아웃페인팅 사용
        "outPaintingParams": {
            "image": input_image_base64,
            "text": prompt,             # AI가 이 프롬프트를 보고 새로운 배경을 그립니다.
            "maskPrompt": mask_prompt,  # [핵심] AI가 이 단어에 해당하는 객체만 '누끼'를 따서 보호합니다.
            "outPaintingMode": outpainting_mode,  
            # "DEFAULT": 객체와 새 배경의 경계선을 부드럽게 섞어줌 (자연스러움)
            # "PRECISE": 객체의 원래 테두리를 칼같이 날카롭게 유지함
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,    # 생성할 이미지 개수
            "quality": "standard",  # 품질
            "width": 512,           # 가로 (64의 배수 권장)
            "height": 512,          # 세로 (64의 배수 권장)
            "cfgScale": 8.0,        # 프롬프트 준수 강도
            "seed": randint(0, 100000), # 다양성을 위한 랜덤 시드
        },
    }
    
    # 3. 부정적 프롬프트가 있으면 바구니에 추가
    if negative_prompt:
        body['outPaintingParams']['negativeText'] = negative_prompt
    
    return json.dumps(body)

def get_response_image(response):
    """
    Bedrock API 응답에서 Base64로 인코딩된 결과 이미지를 추출하여 BytesIO 객체로 복원합니다.
    """
    response = json.loads(response.get('body').read())
    images = response.get('images')
    image_data = base64.b64decode(images[0])

    return BytesIO(image_data)

def get_image_from_model(prompt_content, image_bytes, mask_prompt=None, negative_prompt=None, outpainting_mode="DEFAULT"):
    """
    Amazon Nova Canvas를 사용하여 배경이 교체된 이미지를 생성하는 실행 함수입니다.
    """
    # AWS 세션 및 Bedrock 클라이언트 초기화
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime', region_name='us-east-1') 
    
    # 요청 보낼 JSON 바디 생성
    body = get_image_background_replacement_request_body(
        prompt=prompt_content, 
        image_bytes=image_bytes, 
        mask_prompt=mask_prompt, 
        negative_prompt=negative_prompt, 
        outpainting_mode=outpainting_mode
    ) 
    
    # 모델 호출
    response = bedrock.invoke_model(
        body=body, 
        modelId='amazon.nova-canvas-v1:0', 
        contentType='application/json', 
        accept='application/json'
    )
    
    # 결과 이미지를 메모리 객체(BytesIO)로 반환
    output = get_response_image(response)
    
    return output