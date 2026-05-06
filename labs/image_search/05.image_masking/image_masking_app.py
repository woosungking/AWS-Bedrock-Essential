import streamlit as st
import image_masking_lib as glib


st.set_page_config(layout="wide", page_title="Image Masking with Amazon Bedrock")

st.title("Image Masking with Amazon Bedrock")


col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original Image")
    uploaded_image_file = st.file_uploader("Choose an image file", type=['png', 'jpg'])
    
    if uploaded_image_file:
        uploaded_image_preview = glib.get_bytesio_from_bytes(uploaded_image_file.getvalue())
        st.image(uploaded_image_preview)
    else:
        st.image("images/living-room.png")
    
    
with col2:
    st.subheader("Masking")
    
    masking_mode = st.radio("Masking Mode", ["Image", "Prompt"], horizontal=True)
    
    if masking_mode == 'Image':
    
        uploaded_mask_file = st.file_uploader("Choose a mask file", type=['png', 'jpg'])
        
        if uploaded_mask_file:
            uploaded_mask_preview = glib.get_bytesio_from_bytes(uploaded_mask_file.getvalue())
            st.image(uploaded_mask_preview)
        else:
            st.image("images/mask.png")
    else:
        mask_prompt = st.text_input("Mask Prompt", help="Describe the area to be masked")
    
        
with col3:
    st.subheader("Generation")
    
    prompt_text = st.text_area("Prompt", height=100, help="Describe what you want to see in the final image")

    painting_mode = st.radio("Painting Mode", ["INPAINTING", "OUTPAINTING"])
    
    generate_button = st.button("Generate", type="primary")

    if generate_button:
        with st.spinner("Generating image..."):
            
            if uploaded_image_file:
                image_bytes = uploaded_image_file.getvalue()
            else:
                image_bytes = glib.get_bytes_from_file("images/living-room.png")
            
            if masking_mode == 'Image':
                if uploaded_mask_file:
                    mask_bytes = uploaded_mask_file.getvalue()
                else:
                    mask_bytes = glib.get_bytes_from_file("images/mask.png")
                
                generated_image = glib.get_image_from_model(
                    prompt_content=prompt_text, 
                    image_bytes=image_bytes,
                    masking_mode=masking_mode,
                    mask_bytes=mask_bytes,
                    painting_mode=painting_mode
                )
            else:
                generated_image = glib.get_image_from_model(
                    prompt_content=prompt_text, 
                    image_bytes=image_bytes,
                    masking_mode=masking_mode,
                    mask_prompt=mask_prompt,
                    painting_mode=painting_mode
                )
            
        
        st.image(generated_image)