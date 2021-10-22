import streamlit as st
from PIL import Image

def contacto():
    st.write('''
    # Contacto
    ''')
    logo_github = Image.open('Contact/GitHub-Mark-32px.png')
    logo_linkedin = Image.open('Contact/LI-in-Bug.png')
    logo_correo = Image.open('Contact/Gmail_2013.png')
    logo_whatsapp = Image.open('Contact/whatsapp-logo-1.png')

    col0, col1, col2 = st.columns([1, 2, 30])
    a = 13
    col1.image(logo_github, width=a)
    col2.write('https://github.com/raaraya1')
    col1.image(logo_linkedin, width=a)
    col2.write('https://www.linkedin.com/in/rodrigo-araya-jimenez/')
    col1.image(logo_correo, width=a)
    col2.write('raaraya1@miuandes.cl')
    col1.image(logo_whatsapp, width=a)
    col2.write('+569 66627482')
