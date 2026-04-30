import boto3

def get_tools():
    tools = [
        {
            "toolSpec": {
                "name": "summarize_email",
                "description": "이메일 내용을 요약합니다.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "이메일 내용을 한 줄 또는 두 줄로 요약"
                            },
                            "escalate_complaint": {
                                "type": "boolean",
                                "description": "즉시 추가 검토가 필요한 심각한 불만인지 여부"
                            },
                            "level_of_concern": {
                                "type": "integer",
                                "description": "내용의 중요도를 1~10 사이로 평가",
                                "minimum": 1,
                                "maximum": 10
                            },
                            "overall_sentiment": {
                                "type": "string",
                                "description": "발신자의 전체적인 감정",
                                "enum": ["Positive", "Neutral", "Negative"]
                            },
                            "supporting_business_unit": {
                                "type": "string",
                                "description": "이 이메일을 처리해야 하는 내부 부서",
                                "enum": ["Sales", "Operations", "Customer Service", "Fund Management"]
                            },
                            "customer_names": {
                                "type": "array",
                                "description": "이메일에 언급된 고객 이름 목록",
                                "items": { "type": "string" }
                            },
                            "sentiment_towards_employees": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "employee_name": {
                                            "type": "string",
                                            "description": "직원 이름"
                                        },
                                        "sentiment": {
                                            "type": "string",
                                            "description": "해당 직원에 대한 감정",
                                            "enum": ["Positive", "Neutral", "Negative"]
                                        }
                                    }
                                }
                            }
                        },
                        "required": [
                            "summary",
                            "escalate_complaint",
                            "overall_sentiment",
                            "supporting_business_unit",
                            "level_of_concern",
                            "customer_names",
                            "sentiment_towards_employees"
                        ]
                    }
                }
            }
        }
    ]

    return tools


def get_json_response(input_content):  # text-to-text client function

    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime')
    
    tool_list = get_tools()
    
    message = {
        "role": "user",
        "content": [
            { "text": f"<content>{input_content}</content>" },
            { "text": "반드시 summarize_email 도구를 사용하여 <content> 태그 안의 내용을 기반으로 JSON 형식의 이메일 요약을 생성하세요." }
        ],
    }
    
    response = bedrock.converse(
        modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
    
    return tool_result_dict
