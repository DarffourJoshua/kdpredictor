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


model = pickle.load(open('ckd_model.pkl', 'rb'))
cols=["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane", "class"]

def main():
	st.title("Kidney Disease Prediction Using Hybrid Model")
	
	html_temp = """
	<div style="background:#025246 ;padding:10px">
	<h2 style="color:white;text-align:center;">KD Prediction App </h2>
	</div>
	"""
	st.markdown(html_temp, unsafe_allow_html = True)
	age = st.text_input("Age", 0)
	bp = st.text_input("Blood Pressure", 0)
	sg = st.selectbox("Specific Gravity", [0, 1.005, 1.010, 1.015, 1.020, 1.025])
	al = st.selectbox("Albumin", [0,1,2,3,4,5])
	su = st.selectbox("Sugar", [0,1,2,3,4,5])
	rbc = st.selectbox("Red Blood Cells", ["", "normal","abnormal"])
	pc = st.selectbox("Pus Cell", ["", "normal","abnormal"])
	pcc = st.selectbox("Pus Cell clumps", ["", "present","notpresent"])
	ba = st.selectbox("Bacteria", ["", "present","notpresent"])
	bgr = st.text_input("Blood Glucose Random", 0)
	bu = st.text_input("Blood Urea", 0)
	sc = st.text_input("Serum Creatinine", 0)
	sod = st.text_input("Sodium", 0)
	pot = st.text_input("Potassium", 0)
	hemo = st.text_input("Hemoglobin", 0)
	pcv = st.text_input("Packed Cell Volume", 0)
	wc = st.text_input("White Blood Cell Count", 0)
	rc = st.text_input("Red Blood Cell Count", 0)
	htn = st.selectbox("Hypertension", ["", "yes","no"])
	dm = st.selectbox("Diabetes Mellitus", ["","yes","no"])
	cad = st.selectbox("Coronary Artery Disease", ["","yes","no"])
	appet = st.selectbox("Appetite", ["","good","poor"])
	pe = st.selectbox("Pedal Edema", ["", "yes","no"])
	ane = st.selectbox("Anemia", ["", "yes","no"])
	
	if st.button("Predict"): 
		features = [age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]
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
            'ane': ane}
		#print(data)
		df=pd.DataFrame([list(data.values())], columns=['age','bp','sg','al','su','rbc','pc', 'pcc','ba','bgr','bu','sc','sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'])

		categorical_cols = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
  
		for cat in categorical_cols:
			for col in df.columns:
				le = LabelEncoder()
				if col == cat:
					le.classes_ =  df[col].unique()
					for unique_item in df[col].unique():
						if unique_item not in le.classes_:
							df[col] = ['Unknown' if x == unique_item else x for x in df[col]]
					df[col] = le.transform(df[col])
                    
		features_list = df.values.tolist()      
		prediction = model.predict(features_list)
    
		output = int(prediction[0])
		if output == 1:
			text = "The patient has Chronic Kidney Disease"
		else:
			text = "The patient does not have Chronic Kidney Disease"
    	
		st.success('{}'.format(text))
      
if __name__=='__main__': 
    main()	

	
	
