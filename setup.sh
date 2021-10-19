mkdir -p ~/.streamlit/

echo "\
[theme]\n\
primaryColor='#4d5cea'\n\
backgroundColor='#000000'\n\
secondaryBackgroundColor='#1c1c1c'\n\
textColor='#ffffff'\n\
font='serif'\n\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
