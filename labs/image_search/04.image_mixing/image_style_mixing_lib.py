import os
import boto3
import json
import base64
from io import BytesIO
from random import randint


#get a BytesIO object from file bytes
def get_bytesio_from_bytes(image_bytes):
    image_io = BytesIO(image_bytes)
    return image_io


#get a base64-encoded string from file bytes
def get_base64_from_bytes(image_bytes):
    resized_io = get_bytesio_from_bytes(image_bytes)
    img_str = base64.b64encode(resized_io.getvalue()).decode("utf-8")
    return img_str


#load the bytes from a file on disk
def get_bytes_from_file(file_path):
    with open(file_path, "rb") as image_file:
        file_bytes = image_file.read()
    return file_bytes
    


#InvokeModel API 호출에 대한 문자열화된 요청 본문을 가져옵니다.
def get_image_variation_request_body(prompt, similarity, image_bytes1 = None, image_bytes2 = None):
    
    input_image1_base64 = get_base64_from_bytes(image_bytes1)
    input_image2_base64 = get_base64_from_bytes(image_bytes2)
    
    body = { #InvokeModel API에 전달할 JSON 페이로드를 생성합니다.
        "taskType": "IMAGE_VARIATION",
        "imageVariationParams": {
            "images": [ input_image1_base64, input_image2_base64 ],  
            "prompt": prompt,  
            "similarityStrength": similarity,
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,  # 생성할 변형 개수
            "quality": "standard",  # 허용되는 값은 “standard” 또는 “premium”입니다.
            "width": 512,
            "height": 512,
            "cfgScale": 8.0,
            "seed": randint(0, 100000),  # 무작위 시드 사용
        },
    }
    
    return json.dumps(body)

