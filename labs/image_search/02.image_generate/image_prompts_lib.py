# 필수 AWS 및 유틸리티 라이브러리 가져오기
import boto3
import json
import base64
from io import BytesIO
from random import randint

def get_image_generation_request_body(prompt, negative_prompt=None):
    """
    Nova Canvas 이미지 생성을 위한 JSON 요청 본문을 준비합니다
    Args:
        prompt (str): 생성할 내용을 설명하는 메인 텍스트 프롬프트
        negative_prompt (str, optional): 이미지에서 피해야 할 내용을 설명하는 텍스트
    Returns:
        str: JSON 형식의 요청 본문
    """
    # 기본 요청 구조 만들기
    body = {
        "taskType": "TEXT_IMAGE",  # 텍스트를 이미지로 생성할 것을 지정합니다.
        "textToImageParams": {
            "text": prompt,  # 기본 프롬프트 텍스트
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,    # 단일 이미지 생성
            "quality": "standard",   # Standard 품질 설정 사용
            "height": 512,          # 이미지 높이(픽셀)
            "width": 512,           # 이미지 너비(픽셀)
            "cfgScale": 8.0,       # 이미지가 프롬프트에 얼마나 가깝게 따르는지 제어합니다.
            "seed": randint(0, 100000),  # 다양성을 위한 랜덤 시드
        },
    }
    
    # 제공된 경우 부정적인 프롬프트 추가
    if negative_prompt:
        body['textToImageParams']['negativeText'] = negative_prompt
    
    # 딕셔너리를 JSON 문자열로 변환
    return json.dumps(body)

def get_response_image(response):
    """
    Nova Canvas 응답을 처리하고 이미지를 추출합니다.
    Args:
        response: Bedrock API의 초기 응답
    Returns:
        BytesIO: Image data as a bytes stream
    """
    # JSON 응답 본문 구문 분석
    response = json.loads(response.get('body').read())
    
    # Base64로 인코딩된 이미지 배열 추출
    images = response.get('images')
    
    # base64에서 첫 번째(그리고 유일한) 이미지를 디코딩합니다.
    #    <배열인 이유는>
    #    - "00사진 여러장 줘" 라고 request를 보내면 그에 따른 응답으로 response가 여러장 옴
    #    - "00 사진 줘" 라고 하면 respons로 1장이 옴
    #           [ 이러한 상황들을 통합하기 위해 images는 배열로 옴 ]
    image_data = base64.b64decode(images[0])

    # 손쉬운 처리를 위해 BytesIO 객체로 반환
    return BytesIO(image_data)

def get_image_from_model(prompt_content, negative_prompt=None):
    """
    Nova Canvas를 사용하여 이미지를 생성하는 메인 함수
    Args:
        prompt_content (str): 원하는 이미지를 설명하는 기본 프롬프트
        negative_prompt (str, optional): 이미지에서 피해야 할 사항
    Returns:
        BytesIO: 바이트 스트림으로 생성된 이미지
    """
    # AWS 세션 및 베드락 클라이언트 생성
    session = boto3.Session()
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    
    # 요청 본문 준비
    body = get_image_generation_request_body(prompt_content, negative_prompt=negative_prompt)
    
    # Bedrock을 통해 Nova Canvas 모델 호출
    response = bedrock.invoke_model(
        body=body,
        modelId="amazon.nova-canvas-v1:0", 
        contentType="application/json",
        accept="application/json"
    )
    
    # Process the response and return image
    output = get_response_image(response)
    
    return output
