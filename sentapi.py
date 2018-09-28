from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

# Load the model
MODEL = joblib.load('credit-rf-v1.0.pkl')
MODEL_LABELS = ['positive', 'negative']

@app.route('/predict')
def predict():
    # Retrieve query parameters related to this request.
    text = request.args.get('text')

    # Our model expects a list of records
    features = [[text]]

    text = pipline(text)


    # Use the model to predict the class
    label_index = MODEL.predict(features)
    label_conf = MODEL.predict_proba(features)

    # Retrieve the iris name that is associated with the predicted class
    label = MODEL_LABELS[label_index[0]]
    # Create and send a response to the API caller

    return jsonify(status='complete', label=label,  label_conf = ''.join(str(label_conf)))



if __name__ == '__main__':
    app.run(debug=True)
