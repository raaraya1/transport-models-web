import streamlit as st

def custom():
    # posicion y color fondo select box
    st.markdown('''
    <style>
    .st-b4 {
        display: -webkit-inline-box;
        background: rgb(52, 52, 52);
    }
    <\style>
    ''', unsafe_allow_html=True)

    # color letra select box
    st.markdown('''
    <style>
    .st-cc {
        color: rgb(255 255 255);
    }
    <\style>
    ''', unsafe_allow_html=True)

    # cambiar color de fondo de selecciones
    st.markdown('''
    <style>
    .st-dc {
        background-color: rgb(52 52 52);
    }
    <\style>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <style>
    .st-cd {
    color: rgb(255 255 255);
    }
    <\style>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <style>
    .st-cz {
        background-color: rgb(52 52 52);
    }
    <\style>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <style>
    .st-cb {
    color: rgb(248 249 251);
    }
    .st-cy {
    background-color: rgb(52 52 52);
    }
    <\style>
    ''', unsafe_allow_html=True)

    # cambiar color de opcion seleccionada
    st.markdown('''
    <style>
    .css-9t20qt {
        display: flex;
        -webkit-box-align: center;
        align-items: center;
        padding-top: 0px;
        padding-bottom: 0px;
        background: rgb(20 20 20);
    }
    <\style>
    ''', unsafe_allow_html=True)

    # ocultar opciones de streamlit
    st.markdown('''
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    ''', unsafe_allow_html=True)

    # Personalizar contenedor (pagina entera)
    st.markdown('''
    <style>
    .css-1e5imcs {
    display: block;
    </style>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <style>
    .css-12oz5g7 {
        flex: 1 1 0%;
        width: 100%;
        padding: 0rem 1rem 10rem;
        max-width: 46rem;
    }
    </style>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <style>
    .row-widget.stRadio {
    background: rgb(52, 52, 52);
    border-radius: 5px;
    }
    </style>
    ''', unsafe_allow_html=True)

    # Perzonalizar contenedor (barra de navegacion)
    st.markdown('''
    <style>
    .css-rncmk8 {
        display: flex;
        flex-wrap: wrap;
        -webkit-box-flex: 1;
        flex-grow: 1;
        --fgp-gap: var(--fgp-gap-container);
        margin-top: var(--fgp-gap);
        margin-right: var(--fgp-gap);
        --fgp-gap-container: calc(var(--fgp-gap-parent,0px) - 1rem) !important;
        background-color: rgb(52, 52, 52);
    }
    </style>
    ''', unsafe_allow_html=True)

    # oscurecer botones
    st.markdown('''
    <style>
    .st-c0 {
    background-color: rgb(52 52 52);
    }
    </style>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <style>
    .st-cd {
    background-color: rgb(52 52 52);
    }
    </style>
    ''', unsafe_allow_html=True)

    # letras blancas st.columns
    st.markdown('''
    <style>
    .css-rncmk8 > * {
    color: rgb(255, 255, 255);
    }
    </style>
    ''', unsafe_allow_html=True)

    # bordes redondeados st.columns
    st.markdown('''
    <style>
    .css-rncmk8 {
    border-radius: 5px;
    }
    </style>
    ''', unsafe_allow_html=True)
