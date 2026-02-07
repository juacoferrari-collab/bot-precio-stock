# Bot Precio Stock

Aplicación para consultar precios y stock desde Google Sheets y exposer UI con Streamlit.

Estructura esperada:

bot-precio-stock/
  app.py
  requirements.txt
  README.md
  .gitignore

  src/
    __init__.py
    config.py
    services/
      __init__.py
      catalog.py
      nlu.py

  .streamlit/
    config.toml

Notas:
- Las credenciales de Google Sheets deben guardarse en `.streamlit/secrets.toml` (no versionar).
