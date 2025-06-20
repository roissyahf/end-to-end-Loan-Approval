import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
print("Current tracking URI:", mlflow.get_tracking_uri())

client = mlflow.MlflowClient()

# List registered models
model_target = client.get_registered_model(name="revamp-loan-approval-model")
print(f"Registered Models: {model_target}")

# Try loading the specific version
model = mlflow.pyfunc.load_model("models:/revamp-loan-approval-model/Production")
print("Model loaded successfully.")