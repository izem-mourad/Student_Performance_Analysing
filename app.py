import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

filename = "Models/modelRFC-StdPerformance.sav" # Model SVM classifier
model = pickle.load(open(filename,'rb'))

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/prediction")
def Prediction():
    return render_template("index.html")

@app.route("/predict",methods=['GET','POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    print('first declared features : ', features)

    lis_state=[] # states binary answer
    lis_parent=[] # education level binary answer
    if(features[-2]==0): lis_state=[1,0,0,0,0]
    elif(features[-2]==1): lis_state=[0,1,0,0,0]
    elif(features[-2]==2): lis_state=[0,0,1,0,0]
    elif(features[-2]==3): lis_state=[0,0,0,1,0]
    elif(features[-2]==4): lis_state=[0,0,0,0,1]

    if(features[-2]==0): lis_parent=[1,0,0,0,0,0]
    elif(features[-2]==1): lis_parent=[0,1,0,0,0,0]
    elif(features[-2]==2): lis_parent=[0,0,1,0,0,0]
    elif(features[-2]==3): lis_parent=[0,0,0,1,0,0]
    elif(features[-2]==4): lis_parent=[0,0,0,0,1,0]
    elif(features[-2]==5): lis_parent=[0,0,0,0,0,1]

    features = features[:3] + lis_parent + lis_state
    print('features : ', features)
    print('lis_state : ', lis_state)
    f_features = [np.array(features)]
    prediction = model.predict(f_features)

    if prediction == 1 : prediction = 'Succeed'
    else : prediction = 'Not Succeed'

    print(prediction)
    # the binary classification [note : our model is weak % True Negative values]
    return render_template('index.html', prediction_text= "The Student will " + prediction)


if __name__ == "__main__":
    app.run(debug=True)



