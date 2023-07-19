from flask import * #Flask, redirect, url_for, render_template, request, flash

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
app.secret_key = 'secret'

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        age = request.form["age"]
        sex = request.form["sex"]
        cp = request.form["chestpain"]
        trestbps = request.form["trestbps"]
        chol = request.form["chol"]
        fbs = request.form["fbs"]
        restecg = request.form["restecg"]
        thalach = request.form["thalach"]
        exang = request.form["exang"]
        oldpeak = request.form["oldpeak"]
        slope = request.form["slope"]
        ca = request.form["ca"]
        thal = request.form["thal"]
        
        df = pd.read_csv("newheart.csv")
        df.head(303)

        cols = [0]
        df.drop(df.columns[cols],axis=1,inplace=True)
        y = df.iloc[:,13].values

        drop = ['fbs', 'restecg', 'trestbps', 'chol', 'age', 'slope','sex']
        df = df.drop(drop, axis = 1)
        x = df.iloc[:,:-1].values
        x=x.T
        y=y.T

        x_test=np.array([cp,thalach,exang, oldpeak,ca, thal])
        x_test=pd.DataFrame(x_test)

        knn = KNeighborsClassifier(n_neighbors = 3)
        knn.fit(x.T, y.T)
        prediction = knn.predict(x_test.T)
        print(prediction)

        flash(prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)