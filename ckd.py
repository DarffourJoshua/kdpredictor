import numpy as np
import pandas as pd
import streamlit as st 
from sklearn.preprocessing import  LabelEncoder
import pickle

st.set_page_config(
   page_title="KDP App",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

# Load the trained model
with open('voting_model.pkl', 'rb') as file:  
    model = pickle.load(file)

with open('encoder.pkl', 'rb') as file:
    enc = pickle.load(file)

with open('normalization_model.pkl', 'rb') as file:
    norm = pickle.load(file)
    
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
    
    #Check if the file is uploaded
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
         # Store the id column temporarily
        if 'id' in df.columns:
            id_column = df['id']
        
        # Drop the id and classification columns
        if 'id' in df.columns and 'classification' in df.columns:
            df.drop(["id", "classification"], axis=1, inplace=True)
            
        for i in range(df.shape[0]):
            if df.iloc[i,19] in [' yes','\tyes']:
                df.iloc[i,19]='yes'
            if df.iloc[i,19]=='\tno':
                df.iloc[i,19]='no'
            if df.iloc[i,20]=='\tno':
                df.iloc[i,20]='no'
            if df.iloc[i,15]=='\t?':
                df.iloc[i,15]=np.nan
            if df.iloc[i,15]=='\t43':
                df.iloc[i,15]='43'
            if df.iloc[i,16]=='\t?':
                df.iloc[i,16]=np.nan
            if df.iloc[i,16]=='\t6200':
                df.iloc[i,16]= '6200'
            if df.iloc[i,16]=='\t8400':
                df.iloc[i,16]= '8400'
            if df.iloc[i,17]=='\t?':
                df.iloc[i,17]=np.nan
                
        df.drop(["rbc", "pc", "pcc", "ba", "cad", "appet", "pe", "ane", "bgr", "bu", "wc", "pot"], axis=1, inplace=True)

        #Seperate the categorical columns from the numerical columns
        cat_cols = [col for col in df.columns if df[col].dtype == "object"]
        num_cols = [col for col in df.columns if df[col].dtype != "object"]

        # filling null values, we will use two methods, random sampling for higher null values and 
        def random_value_imputation(feature):
            random_sample = df[feature].dropna().sample(df[feature].isna().sum())
            random_sample.index = df[df[feature].isnull()].index
            df.loc[df[feature].isnull(), feature] = random_sample
            
        # mean/mode sampling for lower null values
        def impute_mode(feature):
            mode = df[feature].mode()[0]
            df[feature] = df[feature].fillna(mode)

        # filling num_cols null values using random sampling method
        for col in num_cols:
            random_value_imputation(col)

        # filling cat_cols null values using mode sampling method
        for col in cat_cols:
            impute_mode(col)
        
        if st.button('Predict', type="secondary", key="predictFile"):
        
            for cat in enc:
                if cat in df.columns:
                    le = LabelEncoder()
                    le.classes_ = np.array(enc[cat] + ['Unknown'])
            
                    # Strip leading whitespace
                    df[cat] = df[cat].str.lstrip()
            
                    # Handle unknown categories by setting them to 'Unknown'
                    df[cat] = df[cat].apply(lambda x: x if x in le.classes_ else 'Unknown')
            
                    # Transform the column
                    df[cat] = le.transform(df[cat])
            
            # Normalize the data
            df_normalized = pd.DataFrame(norm.transform(df), columns=df.columns) 
                              
            #read the columns from the dataframe to the model and create a new column for the prediction
            predictions = model.predict(df_normalized)
            df['Prediction'] = predictions
            df['Prediction'] = df['Prediction'].map({1: 'yes', 0: 'no'})
            
            # Add the id column back to the DataFrame
            df.insert(0, 'id', id_column)
            
        if st.button('Save', key="saveFile"):
            df.to_csv('prediction.csv', index=False)

        

        st.write(df)
    
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
    
   
    
    if st.button("Predict", key="predict"):
        # Convert data to DataFrame
         
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

        
        df=pd.DataFrame([list(data.values())], columns= cols)
        
        cat_cols = [col for col in df.columns if df[col].dtype == 'object']

        
       
        for cat in enc:
            if cat in df.columns:
                le = LabelEncoder()
                le.classes_ = np.array(enc[cat] + ['Unknown'])
        
                # Strip leading whitespace
                df[cat] = df[cat].str.lstrip()
        
                # Handle unknown categories by setting them to 'Unknown'
                df[cat] = df[cat].apply(lambda x: x if x in le.classes_ else 'Unknown')
        
                # Transform the column
                df[cat] = le.transform(df[cat])
                
        # Normalize the data
        df_normalized = pd.DataFrame(norm.transform(df), columns=df.columns) 
        
                                   
        prediction = model.predict(df_normalized)
        output = int(prediction[0])
            # Display prediction result
        if output == 1:
                st.write("Positive")
        else:
                st.write("Negative")



if __name__ == '__main__':
    main()