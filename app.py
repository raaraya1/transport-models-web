import streamlit as st
from ProgramacionLineal.mapas import *
from ProgramacionLineal.Colaboraty_transport_st import *
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

# agregar google tag manager (en head)
tag_code_head = '''
<script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','GTM-5HBDFMT');</script>
'''

# buscar el archivo index.html
path_ind = os.path.dirname(st.__file__)+'/static/index.html'

# abrimos el archivo
with open(path_ind, 'r') as index_file:
    data=index_file.read()

    # verificamos si existe el GA script
    if len(re.findall('UA-', data))==0:

        # ahora insertmas el google analytics y tag manager
        with open(path_ind, 'w') as index_file_f:

            # pegamos los codigos en el archivo HTML file
            newdata1=re.sub('<head>','<head>'+ anlytcs_code, data)
            index_file_f.write(newdata1)
