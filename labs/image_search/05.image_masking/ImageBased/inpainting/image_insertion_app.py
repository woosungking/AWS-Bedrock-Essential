import streamlit as st
import image_insertion_lib as glib


st.set_page_config(layout='wide', page_title='Image Insertion')

st.title('Image Insertion')

col1, col2, col3 = st.columns(3)

placement_options_dict = { #이미지 삽입을 위한 마스크 영역 구성
    'Wall behind desk': (3, 3, 506, 137), #x, y, width, height
    'On top of desk': (78, 60, 359, 115),
    'Beneath desk': (108, 237, 295, 239),
    'Custom': (0, 0, 200, 100), 
}

placement_options = list(placement_options_dict)


with col1:
    st.subheader('Input Image')
    
    uploaded_file = st.file_uploader('Choose an image file', type=['png', 'jpg'])
    
    if uploaded_file:
        uploaded_image_preview = glib.get_bytesio_from_bytes(uploaded_file.getvalue())
        st.image(uploaded_image_preview)
    else:
        st.image('desk.jpg')
    

with col2:
    st.subheader('Insertion Options')
    
    placement_area = st.radio('Select area to insert content', 
        placement_options,
    )
    
    with st.expander('Custom mask dimensions', expanded=False):
        
        mask_dimensions = placement_options_dict[placement_area]
    
        mask_x = st.number_input('X coordinate', value=mask_dimensions[0])
        mask_y = st.number_input('Y coordinate', value=mask_dimensions[1])
        mask_width = st.number_input('Width', value=mask_dimensions[2])
        mask_height = st.number_input('Height', value=mask_dimensions[3])
    
    prompt_text = st.text_area('Describe what to insert', height=100, help='Describe what you want to insert into the image')
    
    generate_button = st.button('Generate Image', type='primary')
    

with col3:
    st.subheader('Generated Image')

    if generate_button:
        with st.spinner('Generating image...'):
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()
            else:
                image_bytes = None
            
            generated_image = glib.get_image_from_model(
                prompt_content=prompt_text, 
                image_bytes=image_bytes, 
                insertion_position=(mask_x, mask_y),
                insertion_dimensions=(mask_width, mask_height),
            )
        
        st.image(generated_image)