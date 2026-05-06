# 필수 라이브러리 임포트
import boto3           # AWS 서비스 호출을 위한 공식 파이썬 SDK
import json            # AI 모델과 통신할 때 사용하는 JSON 데이터 처리
import base64          # 이미지 데이터를 텍스트(문자열)로 변환/복원하기 위한 모듈
from PIL import Image  # 파이썬 이미지 처리 라이브러리 (해상도 확인용)
from io import BytesIO # 하드디스크에 저장하지 않고 메모리상에서 파일을 다루기 위한 모듈
from random import randint

# ------------------------------------------------------------------------
# [유틸리티 함수] 이미지 데이터를 변환하는 헬퍼 함수들
# ------------------------------------------------------------------------

def get_bytesio_from_bytes(image_bytes):
    """
    파일 바이트 데이터(0과 1의 이진 데이터)를 메모리에서 다룰 수 있는 BytesIO 가상 파일 객체로 변환합니다.
    """
    image_io = BytesIO(image_bytes)
    return image_io

def get_base64_from_bytes(image_bytes):
    """
    바이트 데이터를 AI 모델이 읽을 수 있는 형태(Base64 인코딩 문자열)로 변환합니다.
    AWS Bedrock API는 이미지 전송 시 반드시 Base64 문자열 포맷을 요구합니다.
    """
    resized_io = get_bytesio_from_bytes(image_bytes)
    # 메모리 객체의 값을 가져와 Base64로 인코딩한 뒤, 일반 문자열(utf-8)로 디코딩
    img_str = base64.b64encode(resized_io.getvalue()).decode("utf-8")
    return img_str

def get_image_from_bytes(image_bytes):
    """
    바이트 데이터를 PIL Image 객체로 변환합니다. 
    이미지의 가로/세로 해상도(Size)를 알아내기 위해 사용됩니다.
    """
    image_io = BytesIO(image_bytes)
    image = Image.open(image_io)
    return image

def get_png_base64(image):
    """
    PIL Image 객체를 PNG 포맷의 Base64 문자열로 강제 변환합니다.
    마스크 이미지 등 특정 포맷이 필요할 때 유용하게 쓰입니다.
    """
    png_io = BytesIO()
    image.save(png_io, format="PNG")
    img_str = base64.b64encode(png_io.getvalue()).decode("utf-8")
    return img_str

def get_bytes_from_file(file_path):
    """
    내 컴퓨터(로컬) 디스크에 있는 이미지 파일을 읽어서 바이트 데이터로 반환합니다.
    """
    with open(file_path, "rb") as image_file: # 'rb' = Read Binary (바이트 모드로 읽기)
        file_bytes = image_file.read()
    return file_bytes

# ------------------------------------------------------------------------
# [메인 로직] API 요청 본문 생성 및 호출
# ------------------------------------------------------------------------

def get_image_masking_request_body(prompt_content, image_bytes, painting_mode, masking_mode, mask_bytes, mask_prompt):
    """
    Nova Canvas 인페인팅/아웃페인팅 작업을 위한 JSON 요청 바디를 조립합니다.
    
    Args:
        prompt_content (str): 수정/추가할 내용 (예: "선글라스를 씌워줘")
        image_bytes (bytes): 변형할 원본 이미지 데이터
        painting_mode (str): "INPAINTING" (내부 수정) 또는 "OUTPAINTING" (외부 확장)
        masking_mode (str): "Image" (마스크 이미지 직접 제공) 또는 "Prompt" (텍스트로 영역 지정)
        mask_bytes (bytes): (masking_mode가 'Image'일 때) 마스크 이미지 데이터
        mask_prompt (str): (masking_mode가 'Prompt'일 때) 마스크 영역을 지정하는 텍스트 (예: "강아지")
    """
    
    # 1. 원본 이미지의 크기를 알아내어 출력 이미지 크기를 똑같이 맞춥니다.
    original_image = get_image_from_bytes(image_bytes)
    target_width, target_height = original_image.size
    
    # 2. 원본 이미지를 Base64 문자열로 변환
    image_base64 = get_base64_from_bytes(image_bytes)
    
    # 3. 기본 뼈대가 되는 JSON 구조 생성
    body = {
        "taskType": painting_mode, # INPAINTING 또는 OUTPAINTING
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "height": target_height, # 원본 이미지 높이 그대로 사용
            "width": target_width,   # 원본 이미지 너비 그대로 사용
            "cfgScale": 8.0,         # 프롬프트 준수 강도 (보통 8.0 권장)
            "seed": randint(0, 100000), 
        },
    }
    
    # 4. 공통 파라미터 (원본 이미지와 명령 프롬프트) 설정
    params = {
        "image": image_base64,
        "text": prompt_content,  
    }
    
    # 5. 마스킹 방식에 따라 파라미터 분기 처리
    if masking_mode == 'Image':
        # 사용자가 직접 마스크 이미지(흰/검)를 제공한 경우
        mask_base64 = get_base64_from_bytes(mask_bytes)
        params['maskImage'] = mask_base64
    else:
        # 사용자가 텍스트로 마스크 영역을 지정한 경우 (스마트 마스킹)
        params['maskPrompt'] = mask_prompt
        
    # 6. 페인팅 모드에 따라 최종 파라미터 위치 지정
    if painting_mode == 'OUTPAINTING':
        params['outPaintingMode'] = 'DEFAULT' # 자연스러운 배경 확장을 유도
        body['outPaintingParams'] = params
    else:
        body['inPaintingParams'] = params
    
    # 딕셔너리를 JSON 문자열로 묶어서 반환
    return json.dumps(body)

def get_response_image(response):
    """
    Bedrock API의 응답에서 Base64로 인코딩된 이미지 데이터를 추출하여, 
    다시 실제 이미지 파일 객체(BytesIO)로 복원합니다.
    """
    # 1. 응답 데이터의 body 부분을 읽어서 JSON 객체로 파싱
    response = json.loads(response.get('body').read())
    
    # 2. 'images' 배열 추출 (Nova 모델은 결과물을 배열 형태로 줍니다)
    images = response.get('images')
    
    # 3. 첫 번째 이미지의 Base64 텍스트를 이진 데이터(Bytes)로 디코딩
    image_data = base64.b64decode(images[0])

    # 4. 쉽게 저장하고 다룰 수 있도록 BytesIO로 감싸서 반환
    return BytesIO(image_data)

def get_image_from_model(prompt_content, image_bytes, painting_mode, masking_mode, mask_bytes=None, mask_prompt=None):
    """
    Amazon Nova Canvas를 호출하여 최종적으로 편집된 이미지를 받아오는 메인 실행 함수입니다.
    """
    # AWS 세션 시작 및 클라이언트 생성
    session = boto3.Session()
    
    # Nova Canvas 모델은 us-east-1 (버지니아 북부) 리전 사용 권장
    bedrock = session.client(service_name='bedrock-runtime', region_name='us-east-1') 
    
    # 요청 보낼 봉투(JSON 바디) 만들기
    body = get_image_masking_request_body(
        prompt_content, image_bytes, painting_mode, masking_mode, mask_bytes, mask_prompt
    )
    
    # Bedrock 서버에 모델 실행 요청 보내기 (invoke_model)
    response = bedrock.invoke_model(
        body=body, 
        modelId="amazon.nova-canvas-v1:0", 
        contentType="application/json", 
        accept="application/json"
    )
    
    # 받은 응답을 우리가 볼 수 있는 이미지 형태로 가공하기
    output = get_response_image(response)
    
    # 수정 완료: outputd -> output 오타 수정
    return output