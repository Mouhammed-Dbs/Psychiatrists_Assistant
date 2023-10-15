# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:19:16 2023

@author: Mouhammed Dbs
"""

from flask import Flask, render_template,request
import numpy as np
from keras.models import load_model
import pandas as pd

model = load_model('model.h5')
data = pd.read_csv('data_en.csv')
kb = pd.read_csv('main_data.csv')
app = Flask(__name__)
app.debug = True


@app.route('/')
def index():
    return render_template('main.html')


@app.route('/send',methods=['POST'])
def getData():
    res = request.json
    # [{'symptom': 'صعوبة في النوم', 'value': 0.54}, {'symptom': 'الحساسية الكبيرة للعوامل الخارجية (الاضواء والاصوات...)', 'value': 0.5}, {'symptom': 'عدم الرغبة في التحدث', 'value': 0.71}]
    l = np.zeros((1, 126))
    X = data.iloc[:,4] #change 4 to 0 for ar symptoms
    symptoms = [s.get('symptom') for s in res]
    symptomsValue = [s.get('value') for s in res]
    for i,el in enumerate(X):
        for j,e in enumerate(symptoms):
            if el == e:
                l[0][i] = symptomsValue[j]
    predictions = model.predict(l)
    classes = np.argpartition(predictions[0], -2)[0:]

    li_name = ['y'+str((classes)[i]) for i in range(len(classes))]
    s_names = []

    list_names = data[data.columns[5]] #change 5 to 2 for ar disorders
    list_y = data[data.columns[3]]
    for count,v in enumerate(classes):
        for i in range(len(list_y)):
            if list_y[i] == li_name[count]:
                c = int(round(predictions[0][classes[count]],2)*100)
                if c > 10:
                    s_names.append(list_names[i]+ ' %' + str(c) + ' ')
    return '\n'.join(s_names)

@app.route('/ques',methods=['POST'])
def suggestQues():
    res = request.json
    #res = {'selected':[{'symptom': 'صعوبة في النوم', 'value': 0.9}, {'symptom': 'الشعور بالتعب بشكل مستمر', 'value': 0.79}],'list_ques':['Does the patiant have sleeping difficulties?']}
    l = np.zeros((1, 126))
    S = data.iloc[:,4].values #change 4 to 0 for ar symptoms
    Q = np.array([ 'Does the patient suffer from ' + s + '?' for s in data.iloc[:,4].values]) #change 4 to 0 for ar symptoms

    symptoms = [s.get('symptom') for s in list(res['selected'])]
    symptomsValue = [s.get('value') for s in list(res['selected'])]
    list_no = [s for s in list(res['list_ques'])]
    for i,el in enumerate(S):
        for j,e in enumerate(symptoms):
            if el == e:
                l[0][i] = symptomsValue[j]

    predictions = model.predict(l)
    classes = np.argpartition(predictions[0], -2)[0:]
    ques = []
    for max_class in np.flip(classes):
        ill = kb.iloc[max_class,:].values
        for i,v in enumerate(S):
            if v in symptoms:
                ill[i] = 0
        for i,v in enumerate(Q):
            if v in list_no:
                ill[i] = 0
        for i,v in enumerate(ill):
            k = np.argmax(ill)
            if ill[k] > 0:
                if Q[k] not in ques:
                    ques.append(Q[k])
                ill[k] = 0
    return '_'.join(ques) + '#' + '_'.join(list_no)

if __name__ == '__main__':
    app.run(debug=True)
    
# flask run -p 8000
