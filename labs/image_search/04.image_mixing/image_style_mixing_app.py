import streamlit as st
import image_style_mixing_lib as glib


st.set_page_config(layout="wide", page_title="이미지 스타일 믹싱")

st.title("이미지 스타일 믹싱")

col1, col2, col3, col4 = st.columns(4)


with col1:
    st.subheader("첫 번째 이미지")
    
    uploaded_file1 = st.file_uploader("첫 번째 이미지 업로드", type=['png', 'jpg'])
    
    if uploaded_file1:
        uploaded_image_preview = glib.get_bytesio_from_bytes(uploaded_file1.getvalue())
        st.image(uploaded_image_preview)
    else:
        st.image("/environment/workshop/sample_images/bedroom.jpg")

with col2:
    st.subheader("두 번째 이미지")
    
    uploaded_file2 = st.file_uploader("두 번째 이미지 업로드", type=['png', 'jpg'])
    
    if uploaded_file2:
        uploaded_image_preview = glib.get_bytesio_from_bytes(uploaded_file2.getvalue())
        st.image(uploaded_image_preview)
    else:
        st.image("/environment/workshop/sample_images/cat.jpg")

with col3:  
    st.subheader("프롬프트 및 설정")

    prompt_text = st.text_input("프롬프트", value="이미지 혼합", help="생성할 이미지 변형을 간단히 설명하세요.")
    similarity_strength = st.slider("유사성 강도", min_value=0.2, max_value=1.0, value=0.9, step=0.1, help="원본 이미지와의 유사성 정도를 설정합니다. 1.0은 가장 유사하고 0.2는 가장 덜 유사합니다.", format='%.1f')

    generate_button = st.button("이미지 생성", type="primary")    

with col4:
    st.subheader("생성된 이미지")

    if generate_button:

        if uploaded_file1:
            image_bytes1 = uploaded_file1.getvalue()
        else:
            image_bytes1 = glib.get_bytes_from_file("/environment/workshop/sample_images/bedroom.jpg")
            
        if uploaded_file2:
            image_bytes2 = uploaded_file2.getvalue()
        else:
            image_bytes2 = glib.get_bytes_from_file("/environment/workshop/sample_images/cat.jpg")
        
        
        with st.spinner("이미지 생성 중..."):
            generated_image = glib.get_image_from_model(
                prompt_content=prompt_text,
                similarity_strength=float(similarity_strength),
                image_bytes1=image_bytes1,
                image_bytes2=image_bytes2,
            )
        
        st.image(generated_image)

