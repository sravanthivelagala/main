import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, static_url_path='/static')
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    y_probabilities_test = model.predict_proba(final_features)
    y_prob_success = y_probabilities_test[:, 1]
    print("final features", final_features)
    print("prediction:", prediction)
    output = round(prediction[0], 2)
    y_prob = round(y_prob_success[0], 3)
    print(output)

    if output == 0:
        return render_template('index.html',
                               prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A BENIGN WITH PROBABILITY VALUE  {}'.format(
                                   y_prob))
    else:
        return render_template('index.html',
                               prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER WITH PROBABILITY VALUE  {}'.format(
                                   y_prob))


if __name__ == "__main__":
    app.run(debug=True)