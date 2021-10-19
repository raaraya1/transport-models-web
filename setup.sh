mkdir -p ~/.streamlit/

echo "
[theme]
primaryColor="#4d5cea"
backgroundColor="#000000"
secondaryBackgroundColor="#1c1c1c"
textColor="#ffffff"
font="serif"
[server]
port = $PORT
enableCORS = false
headless = true
" > ~/.streamlit/config.toml
