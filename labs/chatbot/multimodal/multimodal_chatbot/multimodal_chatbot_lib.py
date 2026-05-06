import boto3
from io import BytesIO

# [설정] 대화 기록을 최대 몇 개까지 기억할지 결정 (비용과 속도를 조절하는 핸들이에요)
MAX_MESSAGES = 20

# ------------------------------------------------------------------------
# [데이터 구조] 메시지 관리 클래스
# ------------------------------------------------------------------------

class ChatMessage(): 
    """
    텍스트와 이미지를 한데 묶어 관리하는 '메시지 가방'입니다.
    누가 말했는지, 어떤 데이터가 들어있는지 체계적으로 보관합니다.
    """
    def __init__(self, role, message_type, text, bytesio=None, image_bytes=None):
        self.role = role           # 'user'(사용자) 또는 'assistant'(AI)
        self.message_type = message_type # 'text'(글) 또는 'image'(사진)
        self.text = text           # 실제 채팅 텍스트 내용
        self.bytesio = bytesio     # 화면(UI) 출력용 데이터
        self.image_bytes = image_bytes # AI 모델(Bedrock) 전송용 실제 데이터
        

# ------------------------------------------------------------------------
# [유틸리티] 이미지 변환 및 파일 처리 도구
# ------------------------------------------------------------------------

def get_bytesio_from_bytes(image_bytes):
    """이진 데이터를 메모리 상의 이미지 객체(BytesIO)로 변환해 화면에 띄우기 좋게 만듭니다."""
    return BytesIO(image_bytes)


def get_bytes_from_file(file_path):
    """내 컴퓨터에 저장된 사진 파일을 읽어서 AI가 읽을 수 있는 바이트 데이터로 가져옵니다."""
    with open(file_path, 'rb') as image_file:
        return image_file.read()


# ------------------------------------------------------------------------
# [데이터 번역기] 우리 방식의 메시지를 AWS API 전용 형식으로 변환
# ------------------------------------------------------------------------

def convert_chat_messages_to_converse_api(chat_messages):
    """
    우리가 보관 중인 'ChatMessage' 목록을 AWS Converse API가 좋아하는 
    표준 JSON 덩어리로 한 줄씩 번역합니다.
    """
    messages = []
    
    for chat_msg in chat_messages:
        if (chat_msg.message_type == 'image'):
            # 사진일 경우: AWS가 정한 'image' 규격에 맞춰 포장
            messages.append({
                "role": "user", # 이미지는 항상 사용자가 보내는 것이므로 role은 user 고정
                "content": [
                    {
                        "image": {
                            "format": "jpeg", # png나 webp도 가능하지만 보통 jpeg가 무난해요
                            "source": {
                                "bytes": chat_msg.image_bytes
                            }
                        }
                    }
                ]
            })
        else:
            # 텍스트일 경우: 'text' 규격에 맞춰 포장
            messages.append({
                "role": chat_msg.role,
                "content": [
                    {
                        "text": chat_msg.text
                    }
                ]
            })
            
    return messages


# ------------------------------------------------------------------------
# [메인 로직] AI 모델과의 대화 및 기억 관리
# ------------------------------------------------------------------------

def chat_with_model(message_history, new_text=None, new_image_bytes=None):
    """
    기존 대화 내용(history)에 새로운 질문을 붙여서 AI에게 보내고, 답변을 받아옵니다.
    이 함수가 실행될수록 'message_history' 장부에 기록이 계속 쌓입니다.
    """
    session = boto3.Session()
    # 최신 모델 지원을 위해 버지니아 북부 리전 고정!
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='ap-northeast-2')
    
    # 1. 사용자의 새로운 입력을 장부에 적습니다. (글인지 사진인지 구분)
    if new_text:
        new_text_message = ChatMessage('user', 'text', text=new_text)
        message_history.append(new_text_message)
        
    elif new_image_bytes:
        image_bytesio = get_bytesio_from_bytes(new_image_bytes)
        new_image_message = ChatMessage('user', 'image', text=None, bytesio=image_bytesio, image_bytes=new_image_bytes)
        message_history.append(new_image_message)
    
    # 2. [두뇌 관리] 대화 기록이 너무 많아지면 오래된 순서대로 2개씩(질문/답변 세트) 삭제합니다.
    # 기억력이 무한대면 비용이 폭탄처럼 나올 수 있어서 적절히 조절해주는 핵심 장치입니다.
    number_of_messages = len(message_history)
    if number_of_messages > MAX_MESSAGES:
        del message_history[0 : (number_of_messages - MAX_MESSAGES) * 2] 
    
    # 3. 전체 대화 장부를 AWS가 이해할 수 있는 언어로 통번역합니다.
    messages = convert_chat_messages_to_converse_api(message_history)
    
    # 4. 드디어 Claude 3.7 모델 호출! 지금까지의 모든 맥락(Context)을 한꺼번에 던집니다.
    response = bedrock.converse(
        modelId="apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=messages,
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0, # 분석과 팩트 중심 답변을 위해 창의성을 0으로 꽉 잡았습니다.
            "topP": 0.9,
            "stopSequences": []
        },
    )
    
    # 5. AI가 대답한 텍스트를 결과물에서 쏙 뽑아냅니다.
    output = response['output']['message']['content'][0]['text']
    
    # 6. AI의 대답도 '기록 장부'에 적어둡니다. 그래야 다음에 "아까 네가 말한 그거..."라고 물을 수 있거든요.
    response_message = ChatMessage('assistant', 'text', output)
    message_history.append(response_message)
    
    return # 함수가 끝나면 message_history는 업데이트된 상태가 됩니다.