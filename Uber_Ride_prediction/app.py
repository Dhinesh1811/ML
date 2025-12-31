import numpy as np
from flask import Flask, render_template, request
import math
import pickle

app = Flask(__name__)

model2 = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    price = int(request.form["Priceperweek"])
    population = int(request.form["Population"])
    income = int(request.form["Monthlyincome"])
    parking = int(request.form["Averageparkingpermonth"])

    
    int_features = [int(i) for i in request.form.values()]
    final_features = np.array(int_features).reshape(1, -1)
    prediction = model2.predict(final_features)
    output = (round(prediction[0],2))
    return render_template("index.html",
        predict_text=f"🚗 Predicted Weekly Rides: {int(output):,}",
        price=price,
        population=int(population),
        income=int(income),
        parking=int(parking)
    )
if __name__ == '__main__':
    app.run(debug=True)