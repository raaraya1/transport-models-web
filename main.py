import streamlit as st
from Colaboraty_transport_st import Colaboratory_Transport_st
from Colaboraty_transport_dinamic_st import Colaboraty_transport_dinamic_st
from Facility_Location_st import Facility_Location_st
from TSP_MTZ_st import TSP_MTZ_st
from PIL import Image
import re


col1, col2 = st.columns([1, 5])

img = Image.open('logo_escalado.png')
col1.image(img)

col2.title('Modelos de Transporte')

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
                                 ['Colaboratory_Transport', 'Dinamic_Colaboratory_Transport', 'Facility_Location', 'TSP_MTZ'])

if model_name == 'Colaboratory_Transport':
    Colaboratory_Transport_st().interactive_model()

elif model_name == 'Dinamic_Colaboratory_Transport':
    Colaboraty_transport_dinamic_st().interactive_model()

elif model_name == 'Facility_Location':
    Facility_Location_st().interactive_model()

elif model_name == 'TSP_MTZ':
    TSP_MTZ_st().interactive_model()

# agregar google analytics
anlytcs_code = """<script async src="https://www.googletagmanager.com/gtag/js?id=UA-210353274-1"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'UA-210353274-1');
</script>"""

# Fetch the path of the index.html file
path_ind = os.path.dirname(st.__file__)+'/static/index.html'

# Open the file
with open(path_ind, 'r') as index_file:
    data=index_file.read()

    # Check whether there is GA script
    if len(re.findall('UA-', data))==0:

        # Insert Script for Google Analytics
        with open(path_ind, 'w') as index_file_f:

            # The Google Analytics script should be pasted in the header of the HTML file
            newdata=re.sub('<head>','<head>'+anlytcs_code,data)

            index_file_f.write(newdata)
