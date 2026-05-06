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
    """파일 바이트 데이터를 메모리에서 다룰 수 있는 BytesIO 객체로 변환"""
    return BytesIO(image_bytes)

def get_png_base64(image):
    """PIL Image 객체를 강제로 PNG 포맷으로 변환한 뒤 Base64 문자열로 인코딩"""
    png_io = BytesIO()
    image.save(png_io, format="PNG")
    return base64.b64encode(png_io.getvalue()).decode("utf-8")

def get_image_from_bytes(image_bytes):
    """바이트 데이터로부터 해상도와 픽셀을 다룰 수 있는 PIL Image 객체 생성"""
    return Image.open(BytesIO(image_bytes))

def get_bytes_from_file(file_path):
    """로컬 디스크에 있는 이미지 파일을 읽어 이진(Bytes) 데이터로 반환"""
    with open(file_path, "rb") as image_file:
        return image_file.read()

# ------------------------------------------------------------------------
# [핵심 로직 1] 확장된 캔버스용 흑백 마스크(도면) 자동 생성
# ------------------------------------------------------------------------

def get_mask_image_base64(target_width, target_height, position, inside_width, inside_height):
    """
    원본 이미지가 들어갈 자리는 '검은색(보호)', 새로 확장된 배경은 '흰색(생성)'으로 마스크를 그립니다.
    """
    # Nova Canvas 규칙: 흰색(255) = AI가 칠할 곳, 검은색(0) = 원본 유지(보호)할 곳
    inside_color_value = (0, 0, 0)       # 안쪽(원본 사진 위치): 보호해야 하므로 검은색
    outside_color_value = (255, 255, 255) # 바깥쪽(확장된 여백): 새로 그려야 하므로 흰색
    
    # 1. 1024x1024 크기의 거대한 하얀색 도화지를 폅니다.
    mask_image = Image.new("RGB", (target_width, target_height), outside_color_value)
    
    # 2. 원본 사진 크기만 한 까만색 사각형을 만듭니다.
    original_image_shape = Image.new("RGB", (inside_width, inside_height), inside_color_value)
    
    # 3. 하얀 도화지 위의 특정 위치(position)에 까만 사각형을 붙입니다.
    mask_image.paste(original_image_shape, position)
    
    # 완성된 마스크를 Base64 텍스트로 변환
    return get_png_base64(mask_image)

# ------------------------------------------------------------------------
# [핵심 로직 2] 이미지 확장 및 API 요청 본문 조립
# ------------------------------------------------------------------------

def get_image_extension_request_body(prompt, input_image_bytes, negative_prompt=None, vertical_alignment=0.5, horizontal_alignment=0.5):
    """
    원본 사진을 더 큰 캔버스에 배치하고, 늘어난 여백을 AI가 채워 넣도록 요청(Outpainting)합니다.
    
    Args:
        vertical_alignment (float): 세로 정렬 (0.0: 위, 0.5: 중앙, 1.0: 아래)
        horizontal_alignment (float): 가로 정렬 (0.0: 왼쪽, 0.5: 중앙, 1.0: 오른쪽)
    """
    # 1. 원본 이미지의 크기를 측정합니다.
    original_image = get_image_from_bytes(input_image_bytes)
    original_width, original_height = original_image.size
    
    # 2. 새롭게 만들 거대한 캔버스의 크기를 설정합니다 (1024x1024 고정)
    target_width = 1024 
    target_height = 1024
    
    # 3. 정렬 비율(alignment)을 계산하여 원본 이미지를 놓을 (X, Y) 좌표를 정합니다.
    position = ( 
        int((target_width - original_width) * horizontal_alignment), 
        int((target_height - original_height) * vertical_alignment),
    )
    
    # 4. [진짜 도화지 세팅] 회색(235,235,235)의 1024 캔버스를 만들고 원본 사진을 붙입니다.
    extended_image = Image.new("RGB", (target_width, target_height), (235, 235, 235))
    extended_image.paste(original_image, position)
    extended_image_base64 = get_png_base64(extended_image)
    
    # 5. [마스크 세팅] 위에서 만든 도화지 형태와 똑같은 구조의 흑백 마스크를 만듭니다.
    mask_image_base64 = get_mask_image_base64(target_width, target_height, position, original_width, original_height)
    
    # 6. JSON 바디 조립 (Nova Canvas 최신 규격 완벽 적용)
    body = { 
        "taskType": "OUTPAINTING",
        "outPaintingParams": {
            "image": extended_image_base64, # 여백이 생긴 회색 캔버스 (원본 포함)
            "maskImage": mask_image_base64, # 여백은 흰색, 원본은 검은색인 마스크
            "text": prompt,                 # 확장된 배경에 그려넣을 내용 
            "outPaintingMode": "DEFAULT",   
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "premium", # 배경 확장은 픽셀이 많이 필요하므로 프리미엄 권장!
            "width": target_width,
            "height": target_height,
            "cfgScale": 8,
            "seed": randint(0, 100000),
        },
    }
    
    if negative_prompt:
        body['outPaintingParams']['negativeText'] = negative_prompt
    
    return json.dumps(body)

def get_response_image(response):
    """Bedrock 응답에서 확장된 결과 이미지를 복원합니다."""
    response = json.loads(response.get('body').read())
    images = response.get('images')
    image_data = base64.b64decode(images[0])
    return BytesIO(image_data)

def get_image_from_model(prompt_content, image_bytes, negative_prompt=None, vertical_alignment=0.5, horizontal_alignment=0.5):
    """이미지 확장(Canvas Extension)을 실행하는 메인 함수입니다."""
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime', region_name='us-east-1')
    
    body = get_image_extension_request_body(
        prompt_content, image_bytes, 
        negative_prompt=negative_prompt, 
        vertical_alignment=vertical_alignment, 
        horizontal_alignment=horizontal_alignment
    )
    
    response = bedrock.invoke_model(
        body=body, 
        modelId="amazon.nova-canvas-v1:0", 
        contentType="application/json", 
        accept="application/json"
    )
    
    return get_response_image(response)