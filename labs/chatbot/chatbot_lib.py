import boto3

MAX_MESSAGES = 20

class ChatMessage(): #이미지 및 텍스트 메시지를 저장할 수 있는 클래스를 만듭니다.
    def __init__(self, role, text):
        self.role = role
        self.text = text

def convert_chat_messages_to_converse_api(chat_messages):
    messages = []
    
    for chat_msg in chat_messages:
        messages.append({
            "role": chat_msg.role,
            "content": [
                {
                    "text": chat_msg.text
                }
            ]
        })
            
    return messages




def chat_with_model(message_history, new_text=None):
    session = boto3.Session()
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='ap-northeast-2') #Bedrock 클라이언트를 생성합니다.
    
    new_text_message = ChatMessage('user', text=new_text)
    message_history.append(new_text_message)
    
    number_of_messages = len(message_history)
    
    if number_of_messages > MAX_MESSAGES:
        del message_history[0 : (number_of_messages - MAX_MESSAGES) * 2] #사용자 및 어시스턴트 응답을 모두 제거했는지 확인합니다.
    
    messages = convert_chat_messages_to_converse_api(message_history)
    
    response = bedrock.converse(
        modelId="apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=messages,
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0,
            "topP": 0.9,
            "stopSequences": []
        },
    )
    
    output = response['output']['message']['content'][0]['text']
    
    response_message = ChatMessage('assistant', output)
    
    message_history.append(response_message)
    
    return

