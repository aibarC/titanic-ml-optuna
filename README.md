### 1) Build the image
docker build -t titanic-streamlit

### 2) Run the container with 
docker run --rm -p 8501:8501 titanic-streamlit
### or
docker compose up --build