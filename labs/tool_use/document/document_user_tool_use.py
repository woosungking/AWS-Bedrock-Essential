import boto3
import json

# ==========================================
# [준비] 클라이언트 함수 (Mock API)
# ==========================================
def get_popular_song(station_name):
    if "K-Pop" in station_name:
        return {"song": "Supernova", "artist": "aespa"}
    else:
        return {"song": "Shape of You", "artist": "Ed Sheeran"}

def main():
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='ap-northeast-2')
    model_id = "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"

    tools = [{
        "toolSpec": {
            "name": "get_popular_song",
            "description": "특정 라디오 방송국에서 가장 인기 있는 노래를 가져옵니다.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "station_name": {"type": "string", "description": "라디오 방송국 이름"}
                    },
                    "required": ["station_name"]
                }
            }
        }
    }]

    user_message = "K-Pop FM 라디오에서 요즘 제일 잘 나가는 노래가 뭐야?"
    messages = [{"role": "user", "content": [{"text": user_message}]}]

    print(f"👤 사용자 질문: {user_message}\n")

    # ==========================================
    # [1차 호출] 모델에게 질문 + 도구 던지기
    # ==========================================
    response = bedrock.converse(
        modelId=model_id,
        messages=messages,
        toolConfig={"tools": tools}
    )

    output_message = response['output']['message']
    messages.append(output_message) 

    # ★ 원문 출력 1: AI가 도구 사용을 요청하는 원문 (Raw JSON)
    print("👇 [원문 보기 1] AI의 도구 사용 요청 응답 (Raw JSON) 👇")
    print(json.dumps(output_message, indent=2, ensure_ascii=False))
    print("-" * 50)

    # 도구 사용 요청 파싱
    tool_use = next((block['toolUse'] for block in output_message['content'] if 'toolUse' in block), None)

    if tool_use:
        tool_name = tool_use['name']
        tool_input = tool_use['input']
        tool_use_id = tool_use['toolUseId']

        # ==========================================
        # [클라이언트 실행] 파이썬 함수 실행
        # ==========================================
        if tool_name == "get_popular_song":
            station = tool_input['station_name']
            tool_result_data = get_popular_song(station)
            
            # 실행 결과를 메시지에 담기
            tool_result_message = {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"json": tool_result_data}]
                        }
                    }
                ]
            }
            messages.append(tool_result_message)

            # ==========================================
            # [2차 호출] 도구 결과를 모델에게 전달
            # ==========================================
            final_response = bedrock.converse(
                modelId=model_id,
                messages=messages,
                toolConfig={"tools": tools}
            )

            final_message = final_response['output']['message']

            # ★ 원문 출력 2: 최종 자연어 답변이 담긴 원문 (Raw JSON)
            print("\n👇 [원문 보기 2] AI의 최종 답변 응답 (Raw JSON) 👇")
            print(json.dumps(final_message, indent=2, ensure_ascii=False))
            print("-" * 50)

if __name__ == "__main__":
    main()
