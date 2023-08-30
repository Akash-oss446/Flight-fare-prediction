import pandas as pd
from flask import Flask, render_template, request
import pickle
import sklearn
from datetime import datetime

app = Flask(__name__)
model = pickle.load(open('flightpr.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        day = request.form["datedep"]
        Journey_Day = int(pd.to_datetime(day, format="%Y-%m-%dT%H:%M").day)
        Journey_Month = int(pd.to_datetime(day, format="%Y-%m-%dT%H:%M").month)
        Dep_hour = int(pd.to_datetime(day, format="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(day, format="%Y-%m-%dT%H:%M").minute)
        date_arr = request.form["datearr"]
        Arrival_Day = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").hour)
        Arrival_Month = int(pd.to_datetime(day, format="%Y-%m-%dT%H:%M").minute)
        dur_hour = abs(Arrival_Day - Dep_hour)
        dur_min = abs(Arrival_Month - Dep_min)
        Total_stops = int(request.form["stops"])
        airline = request.form['airline']
        if(airline=='Jet Airways'):
            Jet_Airways = 1
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara=0
            Vistara_Premium_economy = 0
            Trujet = 0
        elif(airline=='IndiGo'):
            Jet_Airways = 0
            IndiGo = 1
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Vistara=0
            Trujet = 0
        elif(airline=='Air India'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 1
            Multiple_carriers = 0
            SpiceJet = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Vistara=0
            Trujet = 0
        elif(airline=='Vistara'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Vistara=1
            Trujet = 0
        elif(airline=='Multiple carriers'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 1
            SpiceJet = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Vistara=0
            Trujet = 0
        elif(airline=='GoAir'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            GoAir = 1
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Vistara=0
            Trujet = 0
        elif(airline=='SpiceJet'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 1
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Vistara=0
            Trujet = 0
        elif(airline=='Multiple carriers Premium economy'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 1
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Vistara=0
            Trujet = 0
        elif(airline=='Jet Airways Business'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 1
            Vistara_Premium_economy = 0
            Vistara=0
            Trujet = 0
        elif(airline=='Vistara Premium economy'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Vistara_Premium_economy=1
            Vistara=0
            Jet_Airways_Business = 0
        elif(airline=='Trujet'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 1
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Vistara=0
            Trujet = 1
        else:    
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Vistara=0
            Trujet = 0
        Source = request.form["source"]
        if(Source == 'Delhi'):
            s_Delhi = 1
            s_Kolkata = 0
            s_Mumbai = 0
            s_Chennai = 0
        elif(Source == 'Kolkata'):
            s_Delhi = 0
            s_Kolkata = 1
            s_Mumbai = 0
            s_Chennai = 0
        elif(Source == 'Mumbai'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Mumbai = 1
            s_Chennai = 0
        elif(Source == 'Chennai'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Mumbai = 0
            s_Chennai = 1
        else:     
             s_Delhi = 0
             s_Kolkata = 0
             s_Mumbai = 0
             s_Chennai = 0

        Destination= request.form["Destination"]
        if(Destination == 'Cochin'):
            d_Cochin = 1
            d_Delhi = 0
            d_New_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0
        elif(Destination == 'Delhi'):
            d_Cochin = 0
            d_Delhi = 1
            d_New_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0
        elif(Destination == 'New_Delhi'):
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 1
            d_Hyderabad = 0
            d_Kolkata = 0
        elif(Destination== 'Hyderabad'):
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 0
            d_Hyderabad = 1
            d_Kolkata = 0
        elif(Destination == 'Kolkata'):
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 1
        else:
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0
        prediction = model.predict([[Total_stops, Journey_Day, Journey_Month, 
                                     Dep_hour,Dep_min, Arrival_Day, Arrival_Month,dur_hour,dur_min, 
                                     Air_India, GoAir, IndiGo,
                                     Jet_Airways, Jet_Airways_Business,
                                     Multiple_carriers,
                                     Multiple_carriers_Premium_economy, SpiceJet,
                                     Trujet,Vistara, Vistara_Premium_economy,
                                     s_Chennai, s_Delhi, s_Kolkata, s_Mumbai,
                                     d_Cochin,d_Delhi,d_New_Delhi,d_Hyderabad,d_Kolkata]])
        output = round(prediction[0],2)
        return render_template("/home.html", prediction_text="Your Flight Price is Rs.{}".format(output)) 

    return render_template("/home.html");   

if __name__ == '__main__':
    app.run(debug=True)