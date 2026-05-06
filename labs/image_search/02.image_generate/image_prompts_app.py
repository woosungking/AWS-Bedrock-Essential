import streamlit as st
import image_prompts_lib as glib

st.set_page_config(layout="wide", page_title="Image Generation with Amazon Bedrock")

st.title("Image Generation with Amazon Bedrock")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    
    prompt_text = st.text_area("Enter your prompt", height=100, help="Describe the image you want to generate")
    negative_prompt = st.text_input("Negative prompt", help="Describe what you don't want in the image")

    generate_button = st.button("Generate Image", type="primary")

with col2:
    st.subheader("Generated Image")

    if generate_button:
        with st.spinner("Generating image..."):
            generated_image = glib.get_image_from_model(
                prompt_content=prompt_text, 
                negative_prompt=negative_prompt,
            )
        
        st.image(generated_image)
