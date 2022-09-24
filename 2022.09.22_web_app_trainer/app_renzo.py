import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import base64
from random import randrange
import pandas as pd
import main
import json
import Exercises.Curls
import Exercises.Squats
import Exercises.Extensions
import Exercises.Crunches
import Exercises.Rows
import Exercises.BenchPress

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

st.set_page_config(
    page_title="STARTER TRAINING -UPC",
    page_icon ="img/upc_logo.png",
)

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_upc=get_base64_of_bin_file('img/upc_logo_50x50.png')
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {{
        width: 336px;        
    }}
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
        width: 336px;
        margin-left: -336px;
        background-color: #00F;
    }}
    [data-testid="stVerticalBlock"] {{
        flex: 0;
        gap: 0;
    }}
    #starter-training{{
        padding: 0;
    }}
    #div-upc{{
        #border: 1px solid #DDDDDD;
        background-image: url("data:image/png;base64,{img_upc}");
        position: fixed !important;
        right: 14px;
        bottom: 14px;
        width: 50px;
        height: 50px;
        background-size: 50px 50px;
    }}
    .css-10trblm{{
        color: #FFF;
        font-size: 40px;
        font-family: 'PROGRESS PERSONAL USE';
        src: url(fonts/ProgressPersonalUse-EaJdz.ttf);       
    }}
    #.main {{
        background: linear-gradient(135deg,#a8e73d,#09e7db,#092de7);
        background-size: 180% 180%;
        animation: gradient-animation 3s ease infinite;
        }}

        @keyframes gradient-animation {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
    .block-container{{
        max-width: 100%;
    }}
    </style>
    """ ,
    unsafe_allow_html=True,
)

st.title('STARTER TRAINING')
st.sidebar.markdown('---')
st.markdown("<div id='div-upc'></span>", unsafe_allow_html=True)
st.sidebar.title('The Training App')
st.sidebar.markdown('---')

app_mode = st.sidebar.selectbox('Choose your training:',
    ['(home)','Glute Bridge','Abs', 'Lunges', 'Push Up', 'Squats']
)

if app_mode =='(home)':
    a=0
elif app_mode =='Glute Bridge':
    st.markdown('### __ABS__')
    st.markdown("<hr/>", unsafe_allow_html=True)

    webcam = st.checkbox('Start Webcam')
    st.markdown("Camera status: "+str(webcam))

    trainer, user = st.columns(2)

    with trainer:        
        st.markdown("TRAINER", unsafe_allow_html=True)
        experto = randrange(3)+1

        video_trainer_file="videos_trainer/Glute Bridges/Glute Bridges"+str(experto)+".mp4"
        coord_video = pd.read_csv("videos_trainer/Glute Bridges/Puntos_Glute_Brigdes"+str(experto)+".csv")
        ruta_costos = pd.read_csv("videos_trainer/Glute Bridges/Costos_Glute Bridge_promedio.csv")

        # st.table(coord_video)

        st.video(video_trainer_file, format="video/mp4", start_time=0)
        # st.table(ruta_costos)
        

    with user:
        st.markdown("YOU", unsafe_allow_html=True)
        if webcam:
            Exercises.Squats.start(2,2)
    st.markdown("<hr/>", unsafe_allow_html=True)

elif app_mode =='Abs':
    a=0
elif app_mode =='Lunges':
    a=0
elif app_mode =='Push Up':
    a=0
elif app_mode =='Squats':
    a=0
else:
    a=0

