import boto3
import pandas as pd
import json

def get_tools():
    tools = [
        {
            "toolSpec": {
                "name": "summarize_email",
                "description": "이메일 내용을 요약하고 감정을 분석합니다.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "escalate_complaint": {
                                "type": "boolean",
                                "description": "즉시 상위 부서로 이관(에스컬레이션)하여 추가 검토를 해야 할 만큼 심각한 불만인지 여부"
                            },
                            "level_of_concern": {
                                "type": "integer",
                                "description": "이메일 내용의 심각도/우려 수준을 1에서 10 사이의 숫자로 평가 (10이 가장 심각함)",
                                "minimum": 1,
                                "maximum": 10
                            },
                            "overall_sentiment": {
                                "type": "string",
                                "description": "발신자의 전반적인 감정 상태",
                                "enum": ["Positive", "Neutral", "Negative"] # 시스템 값(enum)은 사내 시스템 연동을 위해 영어를 유지하는 것이 일반적입니다.
                            },
                            "supporting_business_unit": {
                                "type": "string",
                                "description": "이 이메일을 처리해야 하는 내부 담당 부서",
                                "enum": ["Sales", "Operations", "Customer Service", "Fund Management"]
                            },
                            "summary": { # 🚨 원본 코드에서는 이 항목이 properties 밖에 있었습니다! 안으로 넣어서 수정 완료.
                                "type": "string",
                                "description": "이메일의 핵심 내용을 1~2줄로 짧게 요약"
                            }
                        },
                        "required": [
                            "escalate_complaint",
                            "level_of_concern",
                            "overall_sentiment",
                            "supporting_business_unit",
                            "summary"
                        ]
                    }
                }
            }
        }
    ]

    return tools

def get_csv_response(input_content): # text-to-text client function

    session = boto3.Session()
    # region_name은 본인 환경에 맞게 추가/수정하세요
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='ap-northeast-2') 
    
    tool_list = get_tools()
    
    message = {
        "role": "user",
        "content": [
            { "text": f"<content>{input_content}</content>" },
            { "text": "반드시 'summarize_email' 도구를 사용하여 <content> 태그 안의 내용을 분석하고 이메일 요약 JSON을 생성하세요." }
        ],
    }
    
    response = bedrock.converse(
        modelId="apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[message],
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0
        },
        toolConfig={
            "tools": tool_list,
            "toolChoice": {
                "tool": {
                    "name": "summarize_email"
                }
            }
        }
    )
    
    response_message = response['output']['message']
    response_content_blocks = response_message['content']
    content_block = next((block for block in response_content_blocks if 'toolUse' in block), None)
    
    tool_use_block = content_block['toolUse']
    tool_result_dict = tool_use_block['input']
    
    # JSON 딕셔너리를 Pandas DataFrame으로 변환
    data_frame = pd.DataFrame.from_dict([tool_result_dict])
    
    # 🚨 한국어 CSV 저장 시 엑셀에서 글자가 깨지지 않도록 'utf-8-sig' 인코딩을 추가했습니다.
    csv = data_frame.to_csv(index=False, encoding='utf-8-sig') 
    
    return data_frame, csv


# ==========================================
# 테스트를 위한 실행 코드
# ==========================================
if __name__ == "__main__":
    sample_email = """
    안녕하세요, 배송 부서 담당자님.
    지난주에 주문한 노트북이 아직도 도착하지 않았습니다. 배송 조회도 안 되고 고객센터는 전화를 안 받네요.
    업무에 큰 지장이 생겨서 너무 화가 납니다. 당장 환불해주시거나 오늘 내로 조치 결과 알려주세요!!
    """
    
    print("메시지를 분석하여 CSV로 변환 중입니다...\n")
    df, csv_data = get_csv_response(sample_email)
    
    print("✅ [1] Pandas DataFrame 결과 (표 형태):")
    print(df)
    
    print("\n" + "="*50 + "\n")
    
    print("✅ [2] CSV 원시 데이터 출력:")
    print(csv_data)
