FROM python:3.12

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

COPY ../../api ./api

WORKDIR /app/api

EXPOSE 8000

CMD ["python", "prometheus_exporter.py"]
