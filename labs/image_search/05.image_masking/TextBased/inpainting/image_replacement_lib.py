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
    """
    파일 바이트 데이터를 메모리에서 다룰 수 있는 BytesIO 객체로 변환합니다.
    """
    image_io = BytesIO(image_bytes)
    return image_io

def get_base64_from_bytes(image_bytes):
    """
    바이트 데이터를 AI 모델 전송 규격인 Base64 인코딩 문자열로 변환합니다.
    """
    resized_io = get_bytesio_from_bytes(image_bytes)
    img_str = base64.b64encode(resized_io.getvalue()).decode("utf-8")
    return img_str

def get_bytes_from_file(file_path):
    """
    로컬 디스크에 있는 이미지 파일을 읽어 이진(Bytes) 데이터로 반환합니다.
    """
    with open(file_path, "rb") as image_file:
        file_bytes = image_file.read()
    return file_bytes

# ------------------------------------------------------------------------
# [메인 로직] API 요청 본문 생성 및 호출
# ------------------------------------------------------------------------

def get_image_inpainting_request_body(prompt, image_bytes=None, mask_prompt=None, negative_prompt=None):
    """
    Nova Canvas 모델에 보낼 인페인팅(객체 수정/삭제) 전용 JSON 요청 바디를 생성합니다.
    
    Args:
        prompt (str): 마스크된 위치에 새롭게 그려넣을 대상 (예: "선글라스")
                      *주의: 값을 비워두면(None) AI가 마스크된 대상을 완전히 지워버립니다!
        image_bytes (bytes): 변형할 원본 이미지 데이터
        mask_prompt (str): 수정하거나 지울 대상을 찾는 AI 마스킹 텍스트 (예: "강아지")
        negative_prompt (str): 이미지에 절대 포함되지 않았으면 하는 요소 (선택사항)
    """
    # 1. 원본 이미지를 Base64 문자열로 변환
    input_image_base64 = get_base64_from_bytes(image_bytes)
    
    # 2. 인페인팅 기본 JSON 뼈대 생성
    body = { 
        "taskType": "INPAINTING",
        "inPaintingParams": {
            "image": input_image_base64,
            "maskPrompt": mask_prompt, # AI가 사진 속에서 이 단어에 해당하는 객체를 스스로 찾아서 선택함
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,    # 생성할 이미지 개수
            "quality": "standard",  # 품질 (standard 또는 premium)
            "width": 512,           # 출력 이미지 가로 크기 (64의 배수 권장)
            "height": 512,          # 출력 이미지 세로 크기 (64의 배수 권장)
            "cfgScale": 8.0,        # 프롬프트를 따르는 엄격함의 정도
            "seed": randint(0, 100000), # 다양성을 위한 랜덤 시드
        },
    }
    
    # 3. [핵심 로직] 삽입 vs 삭제 분기 처리
    if prompt:  
        # 프롬프트가 있으면: 선택된 영역(maskPrompt)을 이 텍스트(prompt) 내용으로 '교체'합니다.
        body['inPaintingParams']['text'] = prompt 
    # else: 
        # 프롬프트가 없으면: 선택된 영역을 배경과 어울리게 감쪽같이 '삭제(제거)'합니다.
        
    # 4. 부정적 프롬프트(Negative Prompt)가 전달된 경우 추가해 줌 (누락된 로직 보완)
    if negative_prompt:
        body['inPaintingParams']['negativeText'] = negative_prompt
    
    return json.dumps(body)

def get_response_image(response):
    """
    Bedrock 응답 데이터에서 Base64 이미지를 추출하여 BytesIO 객체로 복원합니다.
    """
    response = json.loads(response.get('body').read())
    images = response.get('images')
    image_data = base64.b64decode(images[0])
    
    return BytesIO(image_data)

def get_image_from_model(prompt_content, image_bytes, mask_prompt=None):
    """
    Amazon Nova Canvas를 사용하여 인페인팅 이미지를 생성하는 실행 함수입니다.
    """
    # AWS 세션 및 Bedrock 클라이언트 초기화 (us-east-1 권장)
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime', region_name='us-east-1')
    
    # 요청 보낼 JSON 바디 생성
    body = get_image_inpainting_request_body(
        prompt=prompt_content, 
        image_bytes=image_bytes, 
        mask_prompt=mask_prompt
    )
    
    # 모델 호출
    response = bedrock.invoke_model(
        body=body, 
        modelId="amazon.nova-canvas-v1:0", 
        contentType="application/json", 
        accept="application/json"
    )
    
    # 결과 이미지를 메모리 객체(BytesIO)로 반환
    output = get_response_image(response)
    
    return output