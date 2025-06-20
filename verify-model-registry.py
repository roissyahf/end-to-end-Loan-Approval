import mlflow

mlflow.set_tracking_uri('https://dagshub.com/${{ secrets.DAGSHUB_USERNAME }}/Loan_Approval_Model.mlflow')
mlflow.pyfunc.load_model('models:/revamp-loan-approval-model/Production')
print('Model loaded successfully.')