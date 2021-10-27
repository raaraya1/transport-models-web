import streamlit as st
from FastaiCourse.cat_vs_dog.cat_vs_dog import *
from FastaiCourse.sentiment_classifier.sent_class import *
import re
import os
from custom_streamlit import custom


def fastai_models():
    # Personalizar pagina
    custom()

    st.title('Modelos del curso de FastAI')

    st.write('''
    ## **Contexto**
    ### En construccion...
    ''')

    st.sidebar.write('**Modelos**')
    model_name = st.sidebar.selectbox('Seleccionar Modelo',
                                     ['cat_vs_dog',
                                     'sentiment_classifier'])

    if model_name == 'cat_vs_dog':
        cat_vs_dog_st().model()

    elif model_name == 'sentiment_classifier':
        sentiment_classifier()

#fastai_models()
