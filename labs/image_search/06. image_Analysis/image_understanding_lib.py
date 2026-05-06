import boto3
from io import BytesIO

# ------------------------------------------------------------------------
# [유틸리티 함수]
# ------------------------------------------------------------------------

def get_bytesio_from_bytes(image_bytes):
    """(참고: 이 코드에서는 실제로 사용되지 않는 함수입니다. 복붙의 흔적!)"""
    image_io = BytesIO(image_bytes)
    return image_io

def get_bytes_from_file(file_path):
    """로컬 디스크에 있는 이미지 파일을 읽어 이진(Bytes) 데이터로 반환합니다."""
    with open(file_path, "rb") as image_file:
        file_bytes = image_file.read()
    return file_bytes

# ------------------------------------------------------------------------
# [메인 로직] Claude 3.7 모델을 사용한 이미지 분석 및 답변 생성
# ------------------------------------------------------------------------

# 주의: 이전 코드의 잔재인 mask_prompt는 여기서 전혀 쓰이지 않으므로 파라미터로 받을 필요가 없습니다.
def get_response_from_model(prompt_content, image_bytes):
    """
    Claude 모델에게 사진을 보여주고, 사진에 대한 질문(텍스트)의 답변을 받아옵니다.
    
    Args:
        prompt_content (str): 사진에 대해 AI에게 물어볼 질문 (예: "이 사진 속에 고양이가 몇 마리 있니?", "이 영수증의 총 금액을 텍스트로 추출해 줘")
        image_bytes (bytes): 분석할 원본 이미지 데이터 (get_bytes_from_file로 읽어온 값)
    """
    session = boto3.Session()
    
    # Bedrock 클라이언트 생성 (리전이 생략되어 있는데, us-east-1이나 us-west-2를 명시하는 것을 권장합니다)
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='ap-northeast-2')
    
    # 1. 최신 Converse API 규격에 맞춘 메시지 구조 조립
    image_message = {
        "role": "user", # 대화의 주체 (사용자)
        "content": [
            { "text": "Image 1:" }, # (선택사항) AI에게 이미지가 첨부됨을 알리는 텍스트
            {
                "image": {
                    "format": "jpeg", # 이미지 포맷 (jpeg, png, webp 등)
                    "source": {
                        "bytes": image_bytes # [핵심] Converse API는 Base64로 바꿀 필요 없이 raw Bytes를 그대로 던져도 됩니다!
                    }
                }
            },
            { "text": prompt_content } # 사용자의 실제 질문 프롬프트
        ],
    }
    
    # 2. Bedrock Converse API 호출 (대화형 모델 전용 최신 API)
    response = bedrock.converse(
        # Claude 3.7 Sonnet의 Cross-region 추론(us.) 모델 ID
        modelId="apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[image_message], # 위에서 만든 메시지 배열을 전달
        inferenceConfig={
            "maxTokens": 2000, # AI가 답변할 수 있는 최대 길이
            "temperature": 0   # 0에 가까울수록 사실적이고 일관된 답변, 1에 가까울수록 창의적인 답변
        },
    )
    
    # 3. 응답 객체에서 AI가 대답한 '텍스트(Text)' 결과물만 쏙 뽑아내기
    output = response['output']['message']['content'][0]['text']
    
    return output # 이전 코드들처럼 이미지가 아니라 '문자열(글)'이 반환됩니다!