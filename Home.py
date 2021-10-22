import streamlit as st
from app.ProgramacionLineal.mapas import *
from ProgramacionLineal.Colaboraty_transport_dinamic_st import *
from ProgramacionLineal.Facility_Location_st import *
from ProgramacionLineal.Colaboraty_transport_st import *
from ProgramacionLineal.TSP_MTZ_st import *
from ProgramacionLineal.TSP_dinamic_st import *
from ProgramacionLineal.main import *

from Home.home import *

from Contact.contacto import *

from PIL import Image
from custom_streamlit import custom


img = Image.open('logo_escalado.png')

st.set_page_config(
    page_title="raaraya1", page_icon=img,)

# Personalizar pagina
custom()

# barra de navegacion
navbar = st.container()
paginas = ['Home', 'Linear Models', 'Contact']
pagina = navbar.radio('', paginas)

# opcions
if pagina == 'Home':
    home()

elif pagina == 'Linear Models':
    lineal_models()

elif pagina == 'Contact':
    contacto()


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
