mkdir -p ~/.streamlit/
echo "[general]
email = \"ho_zheng_ting@yahoo.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
