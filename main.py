import streamlit as st
from Colaboraty_transport_st import Colaboratory_Transport_st
from Colaboraty_transport_dinamic_st import Colaboraty_transport_dinamic_st
from Facility_Location_st import Facility_Location_st
from TSP_MTZ_st import TSP_MTZ_st
from PIL import Image



col1, col2 = st.columns([1, 5])

img = Image.open('logo_escalado.png')
col1.image(img)

col2.title('Modelos de Transporte')

st.write('''
## **Contexto**
Este problema surgio de la conversacion entre un grupo de amigos, y este consistia principalmente en organizar las idas y llegadas al colegio, debido a que, con el sistema actual, colapsaba el numero de estacionamientos en el recinto.

Asi, durante la conversacion, se originaron distintas estrategias de como abordar el problema, todas ellas planteando distintos enfoques de modelos de optimizacion.

1. **Modelo con sistema de turnos y apoyo entre vecinos**
2. **Modelo con buses de acercamiento en puntos especificos**
3. **Modelo con buses de acercamiento a domicilio**

''')


st.sidebar.write('**Modelos**')
model_name = st.sidebar.selectbox('Seleccionar Modelo',
                                 ['Colaboratory_Transport', 'Dinamic_Colaboratory_Transport', 'Facility_Location', 'TSP_MTZ'])

if model_name == 'Colaboratory_Transport':
    Colaboratory_Transport_st().interactive_model()

elif model_name == 'Dinamic_Colaboratory_Transport':
    Colaboraty_transport_dinamic_st().interactive_model()

elif model_name == 'Facility_Location':
    Facility_Location_st().interactive_model()

elif model_name == 'TSP_MTZ':
    TSP_MTZ_st().interactive_model()
