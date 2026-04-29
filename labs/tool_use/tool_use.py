import boto3
import math
import json

# 1. 실제 실행될 함수
def cosine_func(x):
    print(f"\n[시스템] 파이썬 cosine_func({x}) 실행 중...")
    return math.cos(math.radians(x))

bedrock = boto3.client(service_name='bedrock-runtime', region_name='ap-northeast-2')
model_id = "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"

# 2. 도구 명세 정의
tool_config = {
    "tools": [{
        "toolSpec": {
            "name": "cosine",
            "description": "Calculate the cosine of a degree value. Use this for any cosine math queries.",
            "inputSchema": {
                "json": {
                    "type": "object", 
                    "properties": {"x": {"type": "number", "description": "The degree value"}}, 
                    "required": ["x"]
                }
            }
        }
    }]
}

# 3. 질문 던지기 (시스템 프롬프트로 도구 사용 강제 유도)
messages = [{"role": "user", "content": [{"text": "반드시 cosine 도구를 호출해서 코사인 60도를 계산해줘."}]}]

# 첫 번째 호출
response = bedrock.converse(modelId=model_id, messages=messages, toolConfig=tool_config)
ai_msg = response['output']['message']
messages.append(ai_msg)

# 4. 도구 호출 여부 확인 및 실행
# content 리스트를 돌면서 'toolUse'가 있는지 확인합니다.
tool_use_block = next((block for block in ai_msg['content'] if 'toolUse' in block), None)

if tool_use_block:
    tool_use = tool_use_block['toolUse']
    query_id = tool_use['toolUseId']
    arg_x = tool_use['input']['x']
    
    # 실제 함수 실행
    result = cosine_func(arg_x)

    # 실행 결과를 메시지에 추가
    messages.append({
        "role": "user",
        "content": [{
            "toolResult": {
                "toolUseId": query_id,
                "content": [{"json": {"result": result}}],
                "status": "success"
            }
        }]
    })

    # 5. 두 번째 호출: 결과를 받은 AI가 최종 답변
    final_response = bedrock.converse(modelId=model_id, messages=messages, toolConfig=tool_config)
    print("\n[AI 최종 답변]:")
    print(final_response['output']['message']['content'][0]['text'])
else:
    # 도구를 안 썼을 때의 예외 출력
    print("\n[AI 답변 (도구 미사용)]:")
    print(ai_msg['content'][0]['text'])
