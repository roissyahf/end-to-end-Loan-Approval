import time
import requests
import logging
from prometheus_exporter import PrometheusMonitor

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

monitor = PrometheusMonitor()

def wait_for_model_ready(api_url: str, timeout: int = 30):
    logger.info("Waiting for model server to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                logger.info("✅ Model server is ready.")
                return True
        except Exception:
            time.sleep(1)
    logger.error("❌ Model server is not responding after timeout.")
    return False

def run_inference(instances):
    input_payload = {"instances": instances}
    monitor.record_batch_size(instances)
    start_time = monitor.record_request_start()
    inference_start = time.time()

    try:
        response = requests.post("http://127.0.0.1:5005/invocations", json=input_payload)
        monitor.record_request_end(start_time)
        monitor.record_inference_latency(inference_start)

        if response.status_code != 200:
            monitor.record_error()
            logger.error(f"❌ Model error: {response.status_code} - {response.text}")
            return None

        result = response.json()
        confidences = result.get("probabilities", [])
        monitor.record_confidences(confidences)

        logger.info("✅ Prediction successful")
        return result

    except Exception as e:
        monitor.record_error()
        monitor.record_request_end(start_time)
        logger.exception("❌ Exception during inference")
        return None

if __name__ == "__main__":
    model_input = {
        "person_age": 35,
        "person_income": 50000,
        "loan_amnt": 100000,
        "loan_int_rate": 7.5,
        "loan_percent_income": 0.2,
        "credit_score": 700,
        "previous_loan_defaults_on_file": "No" # model pipeline has addressed the preprocessing
    }

    test_batch = [model_input]
    model_url = "http://127.0.0.1:5005/ping"

    if wait_for_model_ready(model_url):
        prediction_result = run_inference(test_batch)
        if prediction_result:
            print("Prediction Result:")
            print(prediction_result)
