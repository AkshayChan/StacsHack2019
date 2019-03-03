from flask import Flask, jsonify, request, render_template
app = Flask(__name__)
import os
import json
import numpy as np
# Use DeepSurv from the repo
import sys
sys.path.append('deepsurv')
import deepsurv
from deepsurv import utils, viz, deep_surv
from deepsurv.deepsurv_logger import DeepSurvLogger, TensorboardLogger

import pandas as pd

import lasagne
import matplotlib
import matplotlib.pyplot as plt

@app.route('/ping', methods=['GET'])
def ping():
    return 'pong'

#HOME PAGE
@app.route('/', methods=['GET'])
def main():
    if request.method == 'GET':
        return render_template('load.html')
        
#heart form
@app.route('/heart', methods=['POST'])
def heart():
        return render_template('index1.html') 
    
#Heart data
@app.route('/heart/data', methods=['POST'])
def heartdata():
    #Get the age first
    age = request.form['age']
    age = int(age)
    
    #Get the gender
    gender = request.form.get('gender')
    gender = str(gender)
    gndr = 0
    if gender == "Male":
        gndr = 0
    else:
        gndr = 1
        
    #Get the heartrate
    HR = request.form['HR']
    HR = int(HR)
    
    #Get the Systolic BP
    SBP = request.form['SBP']
    SBP = int(SBP)
    
    #Get the Diastolic BP
    DBP = request.form['DBP']
    DBP = int(DBP)
    
    #Get the Body Mass Index
    BMI = request.form['BMI']
    BMI = int(BMI)
    
    #Get the history of cardiovascular disease
    cvd = request.form.get('cvd')
    cvd = str(cvd)
    cvdint = 0
    if cvd == "No":
        cvdint = 0
    else:
        cvdint = 1
        
    #Get the Atrial Fibriliation
    fab = request.form.get('fab')
    fab = str(fab)
    fabint = 0
    if fab == "No":
        fabint = 0
    else:
        fabint = 1
    
    #Get the Cardiogenic Shock
    sho = request.form.get('sho')
    sho = str(sho)
    shoint = 0
    if sho == "No":
        shoint = 0
    else:
        shoint = 1    
        
    #Get the Congestive Heart Complications
    chf = request.form.get('chf')
    chf = str(chf)
    chfint = 0
    if chf == "No":
        chfint = 0
    else:
        chfint = 1
        
    #Get the Complete Heart Block
    av3 = request.form.get('av3')
    av3 = str(av3)
    av3int = 0
    if av3 == "No":
        av3int = 0
    else:
        av3int = 1
        
    #Get the Myocardial Infarction Order
    miord = request.form.get('miord')
    miord = str(miord)
    miordint = 0
    if miord == "First":
        miordint = 0
    else:
        miordint = 1
        
    #Get the Myocardial Infarction Type
    mitype = request.form.get('miord')
    mitype = str(miord)
    mitypeint = 0
    if miord == "non Q-wave":
        mitypeint = 0
    else:
        mitypeint = 1
        
    
    arr = np.array([250, age, gndr, HR, SBP, DBP, BMI, cvdint, fabint, shoint, chfint, av3int, miordint, mitypeint, 6, 2, 0], dtype=np.float32)
    arr = arr.reshape((1,17))
    model = deep_surv.load_model_from_json("heartModel/model.json", "heartModel/weights")
    risk = model.predict_risk(arr)
    treatment_rec = model.recommend_treatment(arr,1,0)
    print(treatment_rec)
    rectreat = 0
    if treatment_rec > 0:
        print("2nd treament")
        rectreat = 2
    else:
        print("1st treatment")
        rectreat = 1
    print(risk)
    
    risk = str(round(risk[0][0], 3))
    rectreat = str(rectreat)
    
    listr =[1,1,1,1,1,1]
    listr = json.dumps(listr)
    text = "The first method is detailed a bypass surgery, vs the second method which is an angioplasty."
    return render_template('home.html', rectreat = rectreat, risk = risk, text = text)
        
    
    #if request.method == 'POST':
        #age = request.form['age']
        #listr =[1,1,1,1,1,1]
        #listr = json.dumps(listr)
        #return render_template('home.html', age=age, listr = listr)
        

#cancer form
@app.route('/cancer', methods=['POST'])
def cancer():
        return render_template('index2.html') 

#cancer data
@app.route('/cancer/data', methods=['POST'])
def cancerdata():
    
    #Get the age first
    age = request.form['age']
    age = int(age)
    
    #Get the menopause
    menopause = request.form.get('menopause')
    menopause = str(menopause)
    mnpint = 0
    if menopause == "No":
        mnpint = 0
    else:
        mnpint = 1
        
    #Get the hormone therapy
    hormone = request.form.get('hormone')
    hormone = str(hormone)
    hrmint = 0
    if hormone == "No":
        hrmint = 0
    else:
        hrmint = 1
        
    #Get the Tumor Size
    size = request.form['size']
    size = int(size)
    
    #Get the Tumor Grade
    grade = request.form['grade']
    grade = int(grade)
    
    #Get the Nodes
    nodes = request.form['nodes']
    nodes = int(nodes)
    
    #Get the progesterone receptors
    prog_recp = request.form['prog_recp']
    prog_recp = int(prog_recp)
    
    #Get the estrogen receptors
    estrg_recp = request.form['estrg_recp']
    estrg_recp = int(estrg_recp)
    
    #Get the Time to recurrence
    rectime = request.form['rectime']
    rectime = int(rectime)
    
    #Get the recurrence censoring
    censrec = request.form.get('censrec')
    censrec = str(censrec)
    censint = 0
    if censrec == "Censored":
        censint = 0
    else:
        censint = 1
    
    arr = np.array([hrmint, grade, mnpint, age, nodes, prog_recp, estrg_recp],dtype=np.float32)
    arr = arr.reshape((1,7))
    
    model = deep_surv.load_model_from_json("cancerModel/model_gbcs_revised.json", "cancerModel/weights_gbcs_revised")
    risk = model.predict_risk(arr)
    treatment_rec = model.recommend_treatment(arr,1,0)
    print(treatment_rec)
    rectreat = 0
    if treatment_rec > 0:
        print("2nd treament")
        rectreat = 2
    else:
        print("1st treatment")
        rectreat = 1
    print(risk)
    
    risk = str(round(risk[0][0], 3))
    rectreat = str(rectreat)
    
    listr =[1,1,1,1,1,1]
    listr = json.dumps(listr)
    text = "The first method involves 6 cycles of 500 mg/m2 cyclophosphamide, 40 mg/m2 methotrexate and 600 mg/m2 fluorouracil on day 1 and day 8 of the treatment, vs the second method that involves 3 cycles"
    return render_template('home.html', rectreat = rectreat, risk = risk, text=text) 


if __name__ == '__main__':
    host = os.getenv('IP', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    app.debug = True
    app.run(host = host, port = port)