# 필수 라이브러리 임포트
import boto3
import json
import base64
from PIL import Image
from io import BytesIO
from random import randint

# ------------------------------------------------------------------------
# [유틸리티 함수] 이미지/바이트 데이터 변환
# ------------------------------------------------------------------------

def get_bytesio_from_bytes(image_bytes):
    """파일 바이트 데이터를 메모리에서 다룰 수 있는 BytesIO 객체로 변환합니다."""
    image_io = BytesIO(image_bytes)
    return image_io

def get_png_base64(image):
    """PIL Image 객체를 강제로 PNG 포맷으로 변환한 뒤 Base64 문자열로 반환합니다."""
    png_io = BytesIO()
    image.save(png_io, format='PNG')
    img_str = base64.b64encode(png_io.getvalue()).decode('utf-8')
    return img_str

def get_image_from_bytes(image_bytes):
    """바이트 데이터로부터 PIL Image 객체를 생성하여 반환합니다."""
    image_io = BytesIO(image_bytes)
    image = Image.open(image_io)
    return image

def get_bytes_from_file(file_path):
    """로컬 디스크에 있는 이미지 파일을 읽어 이진(Bytes) 데이터로 반환합니다."""
    with open(file_path, "rb") as image_file:
        file_bytes = image_file.read()
    return file_bytes

# ------------------------------------------------------------------------
# [핵심 로직] 파이썬으로 정확한 좌표의 마스크 이미지 직접 그리기
# ------------------------------------------------------------------------

def get_mask_image_base64(target_width, target_height, position, inside_width, inside_height):
    """
    지정된 크기와 위치에 사각형 모양의 마스크(도면)를 파이썬으로 직접 그립니다.
    
    Args:
        target_width, target_height: 전체 원본 이미지의 가로/세로 (도화지 크기)
        position: 마스크를 그릴 시작 (x, y) 좌표
        inside_width, inside_height: 그려 넣을 마스크(사각형)의 가로/세로 크기
    """
    # [수정됨] Nova Canvas는 '흰색' 영역을 수정하고, '검은색' 영역을 보호합니다!
    inside_color_value = (255, 255, 255)  # 흰색: AI가 새롭게 그림을 그릴 영역 (수정)
    outside_color_value = (0, 0, 0)       # 검은색: 원본을 그대로 유지할 영역 (수정)
    
    # 1. 전체 도화지를 까맣게(보호) 칠합니다.
    mask_image = Image.new('RGB', (target_width, target_height), outside_color_value)
    
    # 2. 합성할 물건이 들어갈 크기만큼 하얀색(수정) 사각형을 만듭니다.
    original_image_shape = Image.new('RGB', (inside_width, inside_height), inside_color_value)
    
    # 3. 까만 도화지 위의 특정 좌표(position)에 하얀 사각형을 딱 붙입니다.
    mask_image.paste(original_image_shape, position)
    
    # 완성된 흑백 마스크 이미지를 Base64 문자열로 변환합니다.
    mask_image_base64 = get_png_base64(mask_image)
    
    # (디버깅 팁: 실제로 마스크가 잘 그려졌는지 보려면 아래 주석을 푸세요)
    # mask_image.save('mask_debug.png') 

    return mask_image_base64

# ------------------------------------------------------------------------
# [메인 로직] API 요청 본문 생성 및 호출
# ------------------------------------------------------------------------

def get_image_insertion_request_body(prompt_content, input_image_bytes, insertion_position, insertion_dimensions):
    """
    Nova Canvas 모델에 보낼 좌표 기반 이미지 삽입(인페인팅) 전용 JSON을 생성합니다.
    """
    # 원본 이미지 객체 생성 및 가로세로 크기 추출
    original_image = get_image_from_bytes(input_image_bytes)
    target_width, target_height = original_image.size
    
    # 안전장치: Nova Canvas 규격에 맞게 64의 배수로 반내림 처리 (에러 방지용)
    target_width = (target_width // 64) * 64
    target_height = (target_height // 64) * 64
    
    input_image_base64 = get_png_base64(original_image)
    inside_width, inside_height = insertion_dimensions
    
    # 파이썬으로 흑백 마스크를 생성해 옵니다.
    mask_image_base64 = get_mask_image_base64(
        target_width, target_height, insertion_position, inside_width, inside_height
    )
    
    # [수정됨] Nova Canvas 공식 규격(CamelCase)으로 바디 구조 변경
    body = { 
        "taskType": "INPAINTING", 
        "inPaintingParams": {
            "image": input_image_base64,
            "maskImage": mask_image_base64, # 텍스트(maskPrompt) 대신 우리가 그린 이미지를 직접 전달
            "text": prompt_content,         # 지정된 흰색 네모 안에 그려넣을 내용
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,  
            "quality": "standard",  
            "height": target_height,
            "width": target_width,
            "cfgScale": 8.0,
            "seed": randint(0, 100000), 
        },
    }
    
    return json.dumps(body)

def get_response_image(response):
    """Bedrock API 응답에서 결과 이미지를 복원합니다."""
    response = json.loads(response.get('body').read())
    images = response.get('images')
    image_data = base64.b64decode(images[0])
    return BytesIO(image_data)

def get_image_from_model(prompt_content, image_bytes, mask_prompt=None, negative_prompt=None, insertion_position=None, insertion_dimensions=None):
    """
    좌표 기반 마스크를 생성하여 이미지를 합성(삽입)하는 메인 실행 함수입니다.
    """
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime', region_name='us-east-1') 
    
    # 업로드된 파일이 없으면 기본 이미지(desk.jpg)를 사용하도록 처리
    if image_bytes is None:
        image_bytes = get_bytes_from_file("images/desk.jpg") 
   
    body = get_image_insertion_request_body(
        prompt_content, image_bytes, insertion_position, insertion_dimensions
    ) 
    
    response = bedrock.invoke_model(
        body=body, 
        modelId="amazon.nova-canvas-v1:0", 
        contentType="application/json", 
        accept="application/json"
    )
    
    output = get_response_image(response)
    
    return output