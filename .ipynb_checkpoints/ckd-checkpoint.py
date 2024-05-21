import numpy as np
import pandas as pd
import streamlit as st 
from sklearn.preprocessing import LabelEncoder
import pickle

st.set_page_config(
   page_title="KDP App",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

# Load the trained model
#model = pickle.load(open('finalized_model.pkl', 'rb'))
with open('finalized_model.pkl', 'rb') as file:  
    model = pickle.load(file)

# Define the column names
cols = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"]

def main():
    st.title("Kidney Disease Prediction Using Hybrid Model")
  
   
        
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">KD Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    # Define input fields
    age = st.text_input("Age", 0)
    bp = st.text_input("Blood Pressure", 0)
    sg = st.selectbox("Specific Gravity", [0, 1.005, 1.010, 1.015, 1.020, 1.025])
    al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
    su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
    rbc = st.selectbox("Red Blood Cells", ["", "normal", "abnormal"])
    pc = st.selectbox("Pus Cell", ["", "normal", "abnormal"])
    pcc = st.selectbox("Pus Cell clumps", ["", "present", "notpresent"])
    ba = st.selectbox("Bacteria", ["", "present", "notpresent"])
    bgr = st.text_input("Blood Glucose Random", 0)
    bu = st.text_input("Blood Urea", 0)
    sc = st.text_input("Serum Creatinine", 0)
    sod = st.text_input("Sodium", 0)
    pot = st.text_input("Potassium", 0)
    hemo = st.text_input("Hemoglobin", 0)
    pcv = st.text_input("Packed Cell Volume", 0)
    wc = st.text_input("White Blood Cell Count", 0)
    rc = st.text_input("Red Blood Cell Count", 0)
    htn = st.selectbox("Hypertension", ["", "yes", "no"])
    dm = st.selectbox("Diabetes Mellitus", ["", "yes", "no"])
    cad = st.selectbox("Coronary Artery Disease", ["", "yes", "no"])
    appet = st.selectbox("Appetite", ["", "good", "poor"])
    pe = st.selectbox("Pedal Edema", ["", "yes", "no"])
    ane = st.selectbox("Anemia", ["", "yes", "no"])
    
    data = {
                'age': int(age), 
                'bp': float(bp), 
                'sg': sg, 
                'al': al, 
                'su': su, 
                'rbc': rbc, 
                'pc': pc,
                'pcc': pcc,
                'ba': ba, 
                'bgr': float(bgr), 
                'bu': float(bu), 
                'sc': float(sc), 
                'sod': float(sod), 
                'pot': float(pot), 
                'hemo': float(hemo), 
                'pcv': float(pcv), 
                'wc': float(wc), 
                'rc': float(rc), 
                'htn': htn, 
                'dm': dm, 
                'cad': cad, 
                'appet': appet, 
                'pe': pe, 
                'ane': ane
    }
    df = pd.DataFrame([data], columns=cols)
    
    if st.button("Predict"):
        
        # Convert data to DataFrame
       # Check if all input fields are filled
            # Convert data to DataFrame
       
        
        le = LabelEncoder()
        cat_cols = [col for col in df.columns if df[col].dtype == 'object']
           # [le.fit_transform(df[col]) for col in cat_cols]
        for col in cat_cols:
            df[col] = le.fit_transform(df[col])
        
        prediction = model.predict(df)
        print(prediction[0])
        
        # Display prediction result
        if prediction[0] == 1:
                st.write("Positive")
        else:
                st.write("Negative")



if __name__ == '__main__':
    main()
