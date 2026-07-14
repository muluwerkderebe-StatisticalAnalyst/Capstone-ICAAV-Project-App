# ICAAV Streamlit App

This repository contains a Python Streamlit application for the ICAAV machine learning dashboard.

## Deployment

This is a Streamlit app, not a static website. GitHub Pages is for static content and therefore expects an `index.html` file. That is why GitHub Pages is asking for `index`.

### Recommended deployment path

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io.
3. Click **New app**.
4. Connect your GitHub repository.
5. Choose the branch (for example `main`) and the app file `app.py`.
6. Deploy.

Streamlit Community Cloud will use `requirements.txt` to install dependencies and run `app.py` as the app entrypoint.

## Repository contents

- `app.py` - primary Streamlit app file
- `Pages/` - multipage app modules
- `requirements.txt` - pinned Python 3.14 compatible dependencies
- `assets/` - images and static assets used by the app
- `data/` - data files used by the app
- `.streamlit/config.toml` - Streamlit runtime configuration
- `.gitignore` - ignores virtual environments and local files

## Notes

- If you want to keep the app simple, do not deploy it with GitHub Pages.
- Use Streamlit Cloud or another server that can run Python apps.
- If you do want a GitHub-hosted static landing page, you can create a separate `index.html` for that purpose, but the app itself still needs Streamlit hosting.
