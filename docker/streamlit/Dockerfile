FROM python:3.12

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Copy Streamlit app
COPY ../../streamlit_app ./streamlit_app

# Set working directory to streamlit_app
WORKDIR /app/streamlit_app

EXPOSE 8501

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_TELEMETRY_ENABLED=false \
    STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]