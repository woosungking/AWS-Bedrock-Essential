import streamlit as st #모든 streamlit 명령을 st 별칭을 통해 사용할 수 있습니다.
import multimodal_chatbot_lib as glib #로컬 라이브러리 스크립트에 대한 참조

st.set_page_config(page_title="Multimodal Chatbot") #HTML title
st.title("Multimodal Chatbot") #page title

if 'chat_history' not in st.session_state: #채팅 기록이 아직 생성되지 않았는지 확인합니다.
    st.session_state.chat_history = [] #채팅 기록 초기화

chat_container = st.container()

input_text = st.chat_input("Chat with your bot here") #display a chat input box

uploaded_file = st.file_uploader("Select an image", type=['png', 'jpg'], label_visibility="collapsed")


col1, col2, col3 = st.columns(3)
with col1:
    upload_image_1 = st.button("Add miniature house image")

with col2:
    upload_image_2 = st.button("Add house and car image")

with col3:
    upload_image_3 = st.button("Add miniature car image")

if upload_image_1:
    image_bytes = glib.get_bytes_from_file("images/minihouse.jpg")
    glib.chat_with_model(message_history=st.session_state.chat_history, new_text=None, new_image_bytes=image_bytes)
    
elif upload_image_2:
    image_bytes = glib.get_bytes_from_file("images/house_and_car.jpg")
    glib.chat_with_model(message_history=st.session_state.chat_history, new_text=None, new_image_bytes=image_bytes)

elif upload_image_3:
    image_bytes = glib.get_bytes_from_file("images/minicar.jpg")
    glib.chat_with_model(message_history=st.session_state.chat_history, new_text=None, new_image_bytes=image_bytes)

elif input_text: #사용자가 채팅 메시지를 제출한 후 이 if 블록의 코드를 실행합니다.
    glib.chat_with_model(message_history=st.session_state.chat_history, new_text=input_text, new_image_bytes=None)

elif uploaded_file:
    image_bytes = uploaded_file.getvalue()
    
    glib.chat_with_model(message_history=st.session_state.chat_history, new_text=None, new_image_bytes=image_bytes)

#채팅 기록 다시 렌더링(Streamlit은 이 스크립트를 다시 실행하므로 이전 채팅 메시지를 보존하려면 이 기능이 필요합니다.)
for message in st.session_state.chat_history: #채팅 기록을 반복합니다.
    with chat_container.chat_message(message.role): #지정된 역할에 대한 채팅 라인을 렌더링하며, with 블록에 포함된 모든 내용을 포함합니다.
        if (message.message_type == 'image'):
            st.image(message.bytesio)
        else:
            st.markdown(message.text) #채팅 콘텐츠 표시