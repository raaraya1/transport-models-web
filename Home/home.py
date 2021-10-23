import streamlit as st

def home():
    st.write('''
    # HOME
    ### **¿Dónde estoy?**
    Hola, si llegaste aquí y no sabes cómo, te cuento que esta página la elaboré
    con el fin de jugar un rato. En este sentido, te comento que he estado
    probando una nueva biblioteca de Python llamada streamlit y, por lo visto,
    esta tiene la potencialidad de ser una herramienta bastante útil a la hora
    de materializar una idea o difundir el conocimiento.

    Así, si algo de aquí te llegase a resultar útil siéntete libre de usarlo,
    modificarlo u optimizarlo a tu gusto. Aquí abajo te dejo algunos ejemplos
    de lo que te puedes encontrar en esta página.

    #### **Linear Models App**
    ''')

    col1, col2 = st.columns(2)
    video_file = open('Home/Linear_models.webm', 'rb')
    video_bytes = video_file.read()

    col1.video(video_bytes)
    col2.write('''
    Esta app está escrita en Python con la ayuda de las bibliotecas de `gurobipy`
    para resolver el modelo matemático, `streamlit` para el funcionamiento de la
    app y `folium` y `openrouteservice` para la generación de los mapas y las rutas.
    ''')
