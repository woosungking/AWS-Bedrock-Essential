
import streamlit as st #모든 streamlit 명령을 "st" 별칭을 통해 사용할 수 있습니다.
import chatbot_lib as glib #로컬 라이브러리 스크립트에 대한 참조


st.set_page_config(page_title="Chatbot") #HTML title
st.title("Chatbot") #page title


if 'chat_history' not in st.session_state: #채팅 기록이 아직 생성되지 않았는지 확인합니다.
    st.session_state.chat_history = [] #채팅 기록 초기화


chat_container = st.container()

input_text = st.chat_input("Chat with the bot") #채팅 입력 상자 표시

if input_text:
    glib.chat_with_model(message_history=st.session_state.chat_history, new_text=input_text)


#채팅 기록 다시 렌더링(Streamlit은 이 스크립트를 다시 실행하므로 이전 채팅 메시지를 보존하려면 이 기능이 필요합니다.)
for message in st.session_state.chat_history: #채팅 기록을 반복합니다.
    with chat_container.chat_message(message.role): #채팅 라인을 렌더링하며, with 블록에 포함된 모든 내용을 포함합니다.
        st.markdown(message.text) #채팅 콘텐츠 표시

