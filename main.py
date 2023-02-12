from flask import Flask, request
import pandas as pd
from sklearn.model_selection import train_test_split
from auto_ml import Predictor

app = Flask(__name__)

@app.route('/automl', methods=['POST'])
def run_automl():
    file = request.files['file']
    df = pd.read_excel(file)
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    automl = Predictor(type_of_estimator='regressor', column_descriptions={})
    automl.train(X_train, y_train)

    y_pred = automl.predict(X_test)
    return {'prediction': y_pred.tolist()}

if __name__ == '__main__':
    app.run(debug=True)
