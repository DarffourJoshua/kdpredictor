import numpy as np
import pandas as pd
import streamlit as st 
import upload_file
import predictor

st.set_page_config(
   page_title="KDP App",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)


    
# Define the column names
cols = ["age", "bp", "sg", "al", "su", "sc", "sod", "hemo", "pcv", "rc", "htn", "dm"]

def main():
    st.title("Kidney Disease Prediction Using Hybrid Model")
  
    
    
    
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">KD Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    st.header('Single File Upload')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", accept_multiple_files=False)
    
    upload_file(uploaded_file)
    
    # Define input fields
    age = st.text_input("Age", 0)
    bp = st.text_input("Blood Pressure", 0)
    sg = st.selectbox("Specific Gravity", [0, 1.005, 1.010, 1.015, 1.020, 1.025])
    al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
    su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
    sc = st.text_input("Serum Creatinine", 0)
    sod = st.text_input("Sodium", 0)
    hemo = st.text_input("Hemoglobin", 0)
    pcv = st.text_input("Packed Cell Volume", 0)
    rc = st.text_input("Red Blood Cell Count", 0)
    htn = st.selectbox("Hypertension", ["", "yes", "no"])
    dm = st.selectbox("Diabetes Mellitus", ["", "yes", "no"])
    
    data = {
        'age': int(age), 
        'bp': float(bp), 
        'sg': sg, 
        'al': al, 
        'su': su,
        'sc': float(sc), 
        'sod': float(sod), 
        'hemo': float(hemo), 
        'pcv': float(pcv),
        'rc': float(rc), 
        'htn': htn, 
        'dm': dm
    }
    
    predictor(data)
    
    
    
    # if st.button("Predict", key="predict"):
    #     # Convert data to DataFrame
         
    #     data = {
    #             'age': int(age), 
    #             'bp': float(bp), 
    #             'sg': sg, 
    #             'al': al, 
    #             'su': su,
    #             'sc': float(sc), 
    #             'sod': float(sod), 
    #             'hemo': float(hemo), 
    #             'pcv': float(pcv),
    #             'rc': float(rc), 
    #             'htn': htn, 
    #             'dm': dm
    #     }

        
       



if __name__ == '__main__':
    main()