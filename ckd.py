import streamlit as st 
from upload_files import upload_file
from GFR import GFR
from diagnosis import predictor
import pickle
# from postReq import post
from reportAI import POST

# Load the trained model
with open('voting_model.pkl', 'rb') as file:  
    model = pickle.load(file)

with open('encoder.pkl', 'rb') as file:
    enc = pickle.load(file)

with open('normalization_model.pkl', 'rb') as file:
    norm = pickle.load(file)


st.set_page_config(
   page_title="KDP App",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)




def main():
    st.title("Chronic Kidney Disease Prediction Using Hybrid Model")

    html_temp = """
    <div style="background:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">
            KD Prediction App 
        </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    st.header('Single File Upload')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", accept_multiple_files=False)
    
    upload_file(uploaded_file, model, norm, enc)
    
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
    gender = st.radio('Sex', ['Male', 'Female'], index=None)
    
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

    
    resultsForm = {
        'classification': predictor(data, enc, norm, model),
        'gfr': GFR(sc, gender, age),
        'bp': bp,
        'age': age,
        'gender': gender
    }
    
    if st.button('Generate report', key='next'):
        # st.write(resultsForm) 
        report=POST(resultsForm)
        
        if report:
            # st.write("Generated Report")
            # st.write(report)
            
            # with open('doctors_report.txt', 'w') as file:
            #     file.write(report)
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name='doctors_report.txt',
                mime='text/plain'
            )
    
        
       



if __name__ == '__main__':
    main()