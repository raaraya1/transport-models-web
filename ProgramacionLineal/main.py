import streamlit as st
from ProgramacionLineal.mapas import *
from ProgramacionLineal.Colaboraty_transport_dinamic_st import *
from ProgramacionLineal.Facility_Location_st import *
from ProgramacionLineal.Colaboraty_transport_st import *
from ProgramacionLineal.TSP_MTZ_st import *
from ProgramacionLineal.TSP_dinamic_st import *
import re
import os
from custom_streamlit import custom


def lineal_models():
    # Personalizar pagina
    custom()

    st.title('Modelos de Transporte')

    st.write('''
    ## **Contexto**
    El problema que se plantea a continuación surgió de la conversación entre un grupo de amigos, y este consistía principalmente en organizar las idas y llegadas al colegio, debido a que, con el sistema actual, se colapsaba el número de estacionamientos en el lugar.

    Así, durante la conversación, se originaron distintas estrategias de cómo abordar el problema, todas ellas planteando distintos enfoques de modelación.

    1. **Modelo con sistema de turnos y apoyo entre vecinos**
    2. **Modelo con buses de acercamiento en puntos específicos (paraderos)**
    3. **Modelo con buses de acercamiento a domicilio**

    ''')

    st.sidebar.write('**Modelos**')
    model_name = st.sidebar.selectbox('Seleccionar Modelo',
                                     ['Colaboratory_Transport',
                                     'Dinamic_Colaboratory_Transport',
                                     'Facility_Location',
                                     'TSP_MTZ',
                                     'TSP_dinamic'])

    if model_name == 'Colaboratory_Transport':
        Colaboratory_Transport_st().interactive_model()

    elif model_name == 'Dinamic_Colaboratory_Transport':
        Colaboraty_transport_dinamic_st().interactive_model()

    elif model_name == 'Facility_Location':
        Facility_Location_st().interactive_model()

    elif model_name == 'TSP_MTZ':
        TSP_MTZ_st().interactive_model()

    elif model_name == 'TSP_dinamic':
        TSP_dinamic_st().interactive_model()

'''
if __name__ == "__main__":
    lineal_models()
'''
