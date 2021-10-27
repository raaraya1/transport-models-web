import streamlit as st
import fastbook
#from fastai.vision.all import *
from fastai.text.all import *
import pathlib
import numpy as np
import googletrans
from googletrans import Translator
import requests
import urllib.request
from FastaiCourse.request_from_drive import *
#notas (necesario instalar estas versiones)
#pip install spacy==2.2.4
#pip install googletrans==4.0.0-rc1

def sentiment_classifier():
    # directorio de la carpeta
    path = Path.cwd()

    file_id = '1YRnsXQwl2scl6H2f4xJlnBogckBGZwhe'
    destination = 'm_sent_class.pkl'
    download_file_from_google_drive(file_id, destination)

    # correcciones de ruta
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    # cargando el modelo
    learn = load_learner(str(path) + "\m_sent_class.pkl")

    # Llamar al traductor
    traductor = Translator()

    # Haciendo la prediccion
    resena = st.text_input('Escribe un comentario')
    if resena:
        resena_eng = traductor.translate(resena).text
        st.write(str(resena_eng))
        prediccion = learn.predict(resena_eng)
        prob = int(np.round(prediccion[2][1]*100, 0))
        #st.write(str(prediccion))
        if prediccion[0] == 'pos':
            st.write(f'Se predice que el comentario es positivo con una probabilidad del {prob}%')
        else:
            st.write(f'Se predice que el comentario es negativo con una probabilidad del {100-prob}%')

#sentiment_classifier()
