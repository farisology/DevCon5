from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

# Load the model
MODEL = joblib.load('credit-rf-v1.0.pkl')
MODEL_LABELS = ['non-default', 'default']

@app.route('/predict')
def predict():
    # Retrieve query parameters related to this request.
    RevolvingUtilizationOfUnsecuredLines = request.args.get('RevolvingUtilizationOfUnsecuredLines')
    age = request.args.get('age')
    DebtRatio = request.args.get('DebtRatio')
    MonthlyIncome = request.args.get('MonthlyIncome')
    NumberOfOpenCreditLinesAndLoans = request.args.get('NumberOfOpenCreditLinesAndLoans')
    NumberOfTimes90DaysLate = request.args.get('NumberOfTimes90DaysLate')
    NumberRealEstateLoansOrLines = request.args.get('NumberRealEstateLoansOrLines')
    NumberOfDependents = request.args.get('NumberOfDependents')

    # Our model expects a list of records
    features = [[RevolvingUtilizationOfUnsecuredLines, age, DebtRatio,
                MonthlyIncome, NumberOfOpenCreditLinesAndLoans,
                NumberOfTimes90DaysLate, NumberRealEstateLoansOrLines,
                NumberOfDependents]]


    # Use the model to predict the class
    label_index = MODEL.predict(features)
    label_conf = MODEL.predict_proba(features)

    # Retrieve the iris name that is associated with the predicted class
    label = MODEL_LABELS[label_index[0]]
    # Create and send a response to the API caller

    return jsonify(status='complete', label=label,  label_conf = ''.join(str(label_conf)))



if __name__ == '__main__':
    app.run(debug=True)
