
import streamlit as st #모든 streamlit 명령을 "st" 별칭을 통해 사용할 수 있습니다.
import rag_lib as glib #로컬 라이브러리 스크립트에 대한 참조

st.set_page_config(page_title="Bedrock RAG Demo") #HTML title
st.title("Bedrock RAG Demo") #page title

input_text = st.text_area("Enter your question:", label_visibility="collapsed") #레이블이 없는 여러 줄 텍스트 상자를 표시합니다.
go_button = st.button("Submit", type="primary") #기본 버튼 표시

if go_button: #버튼이 클릭될 때 이 if 블록의 코드가 실행됩니다.
    
    with st.spinner("Working..."): #이 블록의 코드가 실행되는 동안 스피너를 표시합니다.
        response_content, search_results = glib.get_rag_response(question=input_text) #지원 라이브러리를 통해 모델을 호출합니다.
        
        st.write(response_content) #응답 콘텐츠 표시
        
        with st.expander("Search Results"):
            st.table(search_results)


