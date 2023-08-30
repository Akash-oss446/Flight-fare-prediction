#!/usr/bin/env python
# coding: utf-8


from sklearn import SVR
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

train_data = pd.read_csv("Desktop/traindataset.csv")


train_data.head()


train_data.info()


train_data['Duration'].value_counts()


train_data.dropna(inplace=True)


train_data.isnull().sum()


# # EDA

# ### From description we can see the date_of_journey is a object data type
#

# ### we have convert this datatype into timestamp


train_data['Journey_day'] = pd.to_datetime(
    train_data.Date_of_Journey, format="%d/%m/%Y").dt.day


train_data['Journey_month'] = pd.to_datetime(
    train_data['Date_of_Journey'], format="%d/%m/%Y").dt.month


train_data.head()


# since we have converted Date_of_Journey column into integers,Now we drop as it is of no use
train_data.drop(["Date_of_Journey"], axis=1, inplace=True)


train_data.head()


# Departure time is when a plane leaves the gate
# Similar to Date_of_Journey we can extract values from Dept_time
# Extracting Hours
train_data['Day_Hour'] = pd.to_datetime(train_data['Dep_Time']).dt.hour
# Extracting minutes
train_data['Day_Minutes'] = pd.to_datetime(train_data['Dep_Time']).dt.minute
# Now we can drop Dep_Time as it is of no use
train_data.drop(["Dep_Time"], axis=1, inplace=True)


train_data.head()


# Arrival time is when a plane pull up to  the gate
# Similar to Date_of_Journey we can extract values from Dept_time
# Extracting Hours
train_data['Arrival_Hour'] = pd.to_datetime(train_data['Arrival_Time']).dt.hour
# Extracting minutes
train_data['Arrival_Minutes'] = pd.to_datetime(
    train_data['Arrival_Time']).dt.minute
# Now we can drop Dep_Time as it is of no use
train_data.drop(["Arrival_Time"], axis=1, inplace=True)


train_data.head(10000)


# Time taken by plane to reach destination is called Duration
# It is the difference between Departure Time and Arrival time

# Assigning and converting Duration into list
duration = list(train_data['Duration'])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + "0m"
        else:
            duration[i] = "0h" + duration[i]
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))
    duration_mins.append((duration[i].split(sep="m")[0].split()[-1]))


train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins


train_data.head()


# Now we can drop Duration as it is of no use
train_data.drop(["Duration"], axis=1, inplace=True)


train_data.head()


# ## Handling Categorical data


train_data['Airline'].value_counts()


# from graph we can see that Jet Airways Bussiness have the highest price
# Apart from the first Airline almost all having similar median

# Airlane vs Price
sns.catplot(y='Price', x='Airline', data=train_data.sort_values(
    'Price', ascending=False), kind="boxen", height=6, aspect=3)
plt.show()


# As Airline is Nominal Categorical data we will perform OneHotEncoding
Airline = train_data[['Airline']]
Airline = pd.get_dummies(Airline, drop_first=True)
Airline.head()


train_data['Source'].value_counts()


# Source vs Price
sns.catplot(y='Price', x='Source', data=train_data.sort_values(
    'Price', ascending=False), kind="boxen", height=6, aspect=3)
plt.show()


# As Source is Nominal Categorical data we will perform OneHotEncoding
Source = train_data[['Source']]
Source = pd.get_dummies(Source, drop_first=True)
Source.head()


train_data['Destination'].value_counts()


# Source vs Price
sns.catplot(y='Price', x='Destination', data=train_data.sort_values(
    'Price', ascending=False), kind="boxen", height=6, aspect=3)
plt.show()


# As Destination is Nominal Categorical data we will perform OneHotEncoding
Destination = train_data[['Destination']]
Destination = pd.get_dummies(Destination, drop_first=True)
Destination.head(10)


train_data['Route']


# Additional_info contains almost 80% no_info
# Route and Total_Stops are related to each other
train_data.drop(["Route", "Additional_Info"], axis=1, inplace=True)


train_data['Total_Stops'].value_counts()


# As this is case of Ordinal Categorical type we perform LabelEncoder
# Here Values are assigned with coressponding keys
train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2,
                   "3 stops": 3, "4 stops": 4}, inplace=True)


train_data.head(67)


# Concatenate dataframe-->train_data+Airline+Source+Destination
data_train = pd.concat([train_data, Airline, Source, Destination], axis=1)


data_train.head()


data_train.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)


data_train.head()


# ## TestData


test_data = pd.read_csv("Documents/Testdata.csv")


test_data.head()


test_data['Journey_day'] = pd.to_datetime(
    test_data.Date_of_Journey, format="%d/%m/%Y").dt.day


test_data['Journey_month'] = pd.to_datetime(
    test_data['Date_of_Journey'], format="%d/%m/%Y").dt.month


train_data.head()


# since we have converted Date_of_Journey column into integers,Now we drop as it is of no use
test_data.drop(["Date_of_Journey"], axis=1, inplace=True)


# Departure time is when a plane leaves the gate
# Similar to Date_of_Journey we can extract values from Dept_time
# Extracting Hours
test_data['Day_Hour'] = pd.to_datetime(test_data['Dep_Time']).dt.hour
# Extracting minutes
test_data['Day_Minutes'] = pd.to_datetime(test_data['Dep_Time']).dt.minute
# Now we can drop Dep_Time as it is of no use
test_data.drop(["Dep_Time"], axis=1, inplace=True)


# test_data.head(10000)


# Time taken by plane to reach destination is called Duration
# It is the difference between Departure Time and Arrival time

# Assigning and converting Duration into list
duration = list(test_data['Duration'])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + "0m"
        else:
            duration[i] = "0h" + duration[i]
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))
    duration_mins.append((duration[i].split(sep="m")[0].split()[-1]))


test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins


test_data.head(10)


# Arrival time is when a plane pull up to  the gate
# Similar to Date_of_Journey we can extract values from Dept_time
# Extracting Hours
test_data['Arrival_Hour'] = pd.to_datetime(test_data['Arrival_Time']).dt.hour
# Extracting minutes
test_data['Arrival_Minutes'] = pd.to_datetime(test_data['Arrival_Time']).dt.minute
# Now we can drop Dep_Time as it is of no use
test_data.drop(["Arrival_Time"], axis=1, inplace=True)


# Now we can drop Duration as it is of no use
test_data.drop(["Duration"], axis=1, inplace=True)


# As Airline is Nominal Categorical data we will perform OneHotEncoding
Airline = test_data[['Airline']]
Airline = pd.get_dummies(Airline, drop_first=True)


# As Source is Nominal Categorical data we will perform OneHotEncoding
Source = test_data[['Source']]
Source = pd.get_dummies(Source, drop_first=True)


# As Destination is Nominal Categorical data we will perform OneHotEncoding
Destination = test_data[['Destination']]
Destination = pd.get_dummies(Destination, drop_first=True)


# Additional_info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis=1, inplace=True)


# As this is case of Ordinal Categorical type we perform LabelEncoder
# Here Values are assigned with coressponding keys
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2,
                  "3 stops": 3, "4 stops": 4}, inplace=True)


# Concatenate dataframe-->train_data+Airline+Source+Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis=1)


data_test.head()


data_test.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)


data_test.head()


# # Feature Selection

# ## Finding out the best feature which contribute and have good relation with target variable.

# ### Following are some of the features selection methods

# ### 1.Heatmap

# ### 2.feature_importa


data_train.shape


data_train.columns


X = data_train.loc[:, ['Total_Stops','Journey_day', 'Journey_month', 'Day_Hour',
                       'Day_Minutes', 'Arrival_Hour', 'Arrival_Minutes', 'Duration_hours',
                       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
                       'Airline_Jet Airways', 'Airline_Jet Airways Business',
                       'Airline_Multiple carriers',
                       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
                       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
                       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
                       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
                       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()


y = data_train.iloc[:, 1]
y.head()


# Finds coorelation between Independent and Dependent attributes
plt.figure(figsize=(18, 18))
sns.heatmap(train_data.corr(), annot=True, cmap="RdYlGn")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


reg_fr = SVR()
reg_fr.fit(X_train, y_train)


y_pred = reg_fr.predict(X_test)


reg_fr.score(X_train, y_train)


reg_fr.score(X_test, y_test)


sns.distplot(y_test-y_pred)
plt.show()

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
pickle.dump(reg_fr,open('flight.pkl','wb'))
flight=pickle.load(open('flight.pkl','rb'))
