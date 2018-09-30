from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)


# load the model
MODEL = joblib.load('CreditScore-gnb-v1.0.pkl')
MODEL_LABELS = ['non-default', 'default']

@app.route('/predict')

def predict():
	#retrive query parameters related to this request.
	RevolvingUtilizationOfUnsecuredLines = request.args.get('RevolvingUtilizationOfUnsecuredLines')
	age = request.args.get('age')
	NumberOfTime30_59DaysPastDueNotWorse = request.args.get('NumberOfTime30-59DaysPastDueNotWorse')
	DebtRatio = request.args.get('DebtRatio')
	MonthlyIncome = request.args.get('MonthlyIncome')
	NumberOfOpenCreditLinesAndLoans = request.args.get('NumberOfOpenCreditLinesAndLoans')
	NumberOfTimes90DaysLate = request.args.get('NumberOfTimes90DaysLate')
	NumberRealEstateLoansOrLines = request.args.get('NumberRealEstateLoansOrLines')
	NumberOfTime60_89DaysPastDueNotWorse = request.args.get('NumberOfTime60-89DaysPastDueNotWorse')
	NumberOfDependents = request.args.get('NumberOfDependents')

	features = [[RevolvingUtilizationOfUnsecuredLines, age, NumberOfTime30_59DaysPastDueNotWorse, 
	DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate,
	NumberRealEstateLoansOrLines, NumberOfTime60_89DaysPastDueNotWorse, NumberOfDependents]
	]

	#predict new coming data
	label_index = MODEL.predict(features)
	# Get the confidence associated with the prediction
	label_conf = MODEL.predict_proba(features)

	# get the name of the label here
	label = MODEL_LABELS[label_index[0]]

	return jsonify(status = 'complete', label = label)


if __name__ == '__main__':
	app.run(debug = True)







