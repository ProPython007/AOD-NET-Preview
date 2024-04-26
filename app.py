import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.utils import custom_object_scope



# Settings:
## Extra CSS:
st.set_page_config(page_title='Fog Dehazer', page_icon=':bar_chart:', layout='wide')
hide_st_style = '''
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_st_style, unsafe_allow_html=True)
st.title('Fog Dehazer')
st.write("#")

IM_SIZE = (720, 1280)



@st.cache_resource
def load_model():
    with custom_object_scope({'TFOpLambda': TFOpLambda}):
        model = tf.keras.models.load_model('aod_net_fog_v2.h5')
    return model


def load_image(img):
    im = Image.open(img)
    new_image = im.resize((1280, 720))
    image = np.array(new_image)
    return image


def get_pred(img):
    timg = tf.convert_to_tensor(img)
    rimg = tf.image.resize(timg, size=IM_SIZE, antialias=True)
    rimg = rimg/255.0
    eimg = tf.expand_dims(rimg, 0)
    pred = model(eimg)
    return pred


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)



model = load_model()
col1, col2 = st.columns(2)

with col1:
    st.subheader('Input Image:')
    uploadFile = st.file_uploader(label=" ", type=['jpg', 'png'])
    if uploadFile is not None:
        img = load_image(uploadFile)
        st.image(img)

with col2:
    st.subheader('Output Image:')
    if uploadFile is not None:
        st.write("##")
        for _ in range(4): st.write("#")
        st.image(tensor_to_image(get_pred(img)))
