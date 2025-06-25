from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import (
    Counter, Histogram, Gauge,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)

class PrometheusMonitor:
    def __init__(self):
        # Use a custom registry to avoid global metric duplication
        self.registry = CollectorRegistry()

        # HTTP Metrics
        self.request_count = Counter('http_requests_total', 'Total HTTP Requests received', registry=self.registry)
        self.request_latency = Histogram('http_request_duration_seconds', 'Full request duration in seconds', registry=self.registry)
        self.active_requests = Gauge('active_requests', 'Number of active inference requests in progress', registry=self.registry)

        # System Metrics
        self.cpu_usage = Gauge('system_cpu_usage', 'CPU Usage Percentage', registry=self.registry)
        self.ram_usage = Gauge('system_ram_usage', 'RAM Usage Percentage', registry=self.registry)

        # Inference Metrics
        self.prediction_latency = Histogram('model_prediction_latency_seconds', 'Inference latency in seconds', registry=self.registry)
        self.model_reloads = Counter('model_reloads_total', 'Number of times the model has been reloaded', registry=self.registry)
        self.error_counter = Counter('model_errors_total', 'Total number of model prediction errors', registry=self.registry)
        self.prediction_confidence = Histogram(
            'model_prediction_confidence',
            'Histogram of prediction confidence values',
            buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            registry=self.registry
        )

        # Initial state
        self.model_reloads.inc()

    def update_system_metrics(self):
        self.cpu_usage.set(psutil.cpu_percent(interval=1))
        self.ram_usage.set(psutil.virtual_memory().percent)

    def record_request_start(self):
        self.request_count.inc()
        self.active_requests.inc()
        return time.time()

    def record_request_end(self, start_time):
        duration = time.time() - start_time
        self.request_latency.observe(duration)
        self.active_requests.dec()

    def record_inference_latency(self, start_time):
        duration = time.time() - start_time
        self.prediction_latency.observe(duration)

    def record_confidences(self, confidences):
        if isinstance(confidences, list):
            for val in confidences:
                if isinstance(val, (float, int)):
                    self.prediction_confidence.observe(val)

    def record_error(self):
        self.error_counter.inc()

    def prometheus_response(self):
        """Return a response suitable for Prometheus scrape."""
        self.update_system_metrics()
        return Response(generate_latest(self.registry), mimetype=CONTENT_TYPE_LATEST)

# ---------------------
# Flask Setup
# ---------------------
app = Flask(__name__)
monitor = PrometheusMonitor()

@app.route('/predict', methods=['POST'])
def predict():
    request_start = monitor.record_request_start()

    try:
        data = request.get_json()

        inference_start = time.time()

        # Model expects {"instances": [...]}
        input_payload = {"instances": [data]}
        response = requests.post("http://127.0.0.1:5005/invocations", json=input_payload)

        monitor.record_inference_latency(inference_start)
        monitor.record_request_end(request_start)

        if response.status_code != 200:
            monitor.record_error()
            return jsonify({"error": response.text}), response.status_code

        result = response.json()
        prediction = result.get("predictions", [None])[0]

        try:
            probabilities = result.get("probabilities", [])
            monitor.record_confidences(probabilities)
        except:
            pass

        return jsonify({"prediction": prediction, "probabilities": probabilities})

    except Exception as e:
        monitor.record_error()
        monitor.record_request_end(request_start)
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify(status="ok"), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
