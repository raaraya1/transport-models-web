import streamlit as st
import fastbook
from fastai.vision.all import *
import pathlib
from PIL import Image
import numpy as np
import requests
import urllib.request
from FastaiCourse.request_from_drive import *
#https://drive.google.com/file/d/1_pyJdn4pIIp5UVU1poidRh0oiE5iTGNq/view?usp=sharing
#1_pyJdn4pIIp5UVU1poidRh0oiE5iTGNq

def is_cat(x):
    return x[0].isupper()

class cat_vs_dog_st():
    def __init__(self):
        pass

    def model(self):
        # para cargar el modelo
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

        path = Path.cwd()
        file_id = '1_pyJdn4pIIp5UVU1poidRh0oiE5iTGNq'
        destination = 'm_cat_vs_dog.plk'
        download_file_from_google_drive(file_id, destination)

        path = Path(str(path) + '\m_cat_vs_dog.plk')
        learn = load_learner(path)

        # Haciendo la prediccion
        archivo = st.file_uploader('Colaca la imagen de un gato o perro')
        if archivo:
            st.image(archivo, width=128)
            img = PILImage.create(archivo)
            prediccion = learn.predict(img)
            prob = np.round(prediccion[2][1]*100, 0)
            if prediccion[0] == 'True':
                st.write(f'Se predice que es un gato con una probabilidad del {prob}%')
            else:
                st.write(f'Se predice que es un perro con una probabilidad del {100-prob}%')
