import pickle
from flask import Flask,request,jsonify,render_template,url_for
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

## import ridge regression and standardscaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standarscaler=pickle.load(open('models/scaler.pkl','rb'))



@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        temp=float(request.form.get('temperature'))
        rh=float(request.form.get('rh'))
        ws=float(request.form.get('ws'))
        rain=float(request.form.get('rain'))
        ffmc=float(request.form.get('ffmc'))
        dmc=float(request.form.get('dmc'))
        isi=float(request.form.get('isi'))
        classes=float(request.form.get('classes'))
        reg=float(request.form.get('region'))
        new_data_scaled=standarscaler.transform([[temp,rh,ws,rain,ffmc,dmc,isi,classes,reg]])
        result=ridge_model.predict(new_data_scaled)
        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')
        








if __name__=="__main__":
    app.run(host="0.0.0.0")