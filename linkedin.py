import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#import, clean data
s = pd.read_csv("social_media_usage.csv")
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return(x)
ss=pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0),
    "marital":np.where(s["marital"] == 1, 1, 0),
    "female": np.where(s['gender'] == 1, 0,
                       np.where(s['gender'] == 2, 1, np.nan)),
    "age":np.where(s["age"] > 98, np.nan, s["age"])
})
ss=ss.dropna()

#set y and x variable 
y=ss["sm_li"]
x=ss[["income","education","parent","marital","female","age"]]

#80/20 split of train and test data 
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=123)

#fit data 
lr = LogisticRegression(class_weight="balanced")
lr.fit(x_train,y_train)


#create app 
st.markdown("# LinkedIn User Prediction")

#create variable to use in new data "answer" to run prediction 
#income selection box 
inc_display = ("Less than $10k","$10k - 20k","$20k - 30k", "$30k - 40k","$40k - 50k",
    "$50k - 75k","$75k - 100k", "$100k - $150k", "Over $150k")
options = list(range(len(inc_display)))
inc_value = 1+st.selectbox("gender", options, format_func=lambda x: inc_display[x])
#education selection box 
edu_display = ("Less than high school","High school incomplete","High school graduate",
    "Some college","Two-year associate degree","Four-year college or university degree","Some postgraduate schooling",
    "Postgraduate or professional degree")
options = list(range(len(edu_display)))
edu_value = 1+st.selectbox("gender", options, format_func=lambda x: edu_display[x])
#radio buttons for parent, marital status, gender 
parentradio = st.radio("Are you a parent?", ("Yes", "No"))
if parentradio == "Yes":
    parent = 1
else:
    parent = 0 
maritalradio = st.radio("Are you married?", ("Yes", "No"))
if maritalradio == "Yes":
    marital = 1
else:
    marital = 0 
genderradio = st.radio("Are you male or female?", ("Male", "Female"))
if genderradio == "Male":
    gender = 0
else:
    gender = 1 
#age slider 
age = st.slider("How old are you?", min_value=18, max_value=98)

#create list with inputs from answers above 
answer = [inc_value,edu_value,parent,marital,gender,age]

#run list in lr prediction 
predicted_class = lr.predict([answer])
probs = lr.predict_proba([answer])

#print results 
if predicted_class == 1:
    result = "Yes"
    st.write("You are predicted to be a LinkedIn user.")
else:
    result = "No"
    st.write("You are not predicted to be a LinkedIn user.")
st.write("The probability you are a LinkedIn user is:", {probs[0][1]})

#create gauge
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = (probs[0][1]),
    title = {'text': f"Likely LinkedIn User? {result}"},
    gauge = {"axis": {"range":[0,1]},
        "steps": [
            {"range":[0,0.5], "color":"red"},
            {"range":[0.5,1], "color":"green"}],
        "bar":{"color":"yellow"}}
))

st.plotly_chart(fig)
