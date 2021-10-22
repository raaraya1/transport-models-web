import streamlit as st

def home():
    st.write('''
    # HOME
    ### **¿Donde estoy?**
    Hola, si llegaste aqui y no sabes como, te cuento que esta pagina la elaboré
    con el fin de jugar un rato. En sentido, te aclaro desde el inicio que no soy
    ningun programador por profesión, sin embargo, si he visto en esta la
    oportunidad para materializar una idea y difundir el conocimiento.


    Así, si algo de aqui te llegase a resultar util sientete libre de
    usarlo, modificarlo u optimizarlo a tu gusto. Aqui abajo te dejo algunos ejemplos
    de lo que te puedes encontrar en esta pagina.


    #### **Linear Models App**
    ''')

    col1, col2 = st.columns(2)
    video_file = open('Home/Linear_models.webm', 'rb')
    video_bytes = video_file.read()

    col1.video(video_bytes)
    col2.write('''
    Esta app esta escrita en Python con la ayuda
    de las bibliotecas de `gurobipy` para resolver el modelo,
    `streamlit` para el funcionamiento de la app y `folium`
    para la generación de los mapas.
    ''')
