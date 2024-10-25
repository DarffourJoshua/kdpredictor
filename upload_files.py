import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import  LabelEncoder
from GFR import GFR
from reportAI import POST
import datetime

def upload_file(file, model, norm, enc):
    #Check if the file is uploaded
    if file is not None:
        df = pd.read_csv(file)
            
        if 'gender' not in df.columns:
            st.error("Expected the gender column in the file")
            st.stop()
        else:
            gender_column = df['gender']
            df.drop(['gender'], axis=1, inplace=True)
      

        
        # Drop the id and classification columns
        if 'id' in df.columns and 'classification' in df.columns:
            df.drop(["id", "classification" ], axis=1, inplace=True)
            

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
        if file is not None:
            if st.button('Next', type="secondary", key="predictFile"):
                for cat in enc:
                    if cat in df.columns:
                        le = LabelEncoder()
                        le.classes_ = np.array(enc[cat] + ['Unknown'])
                
                        # Strip leading whitespace
                        # df[cat] = df[cat].str.lstrip()
                        if df[cat].dtype == 'object':
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
                df['Prediction'] = df['Prediction'].map({1: 'positive', 0: 'negative'})
                
                # store the user data
                resultsForms = {
                    'age': [age for age in df['age']],
                    'bp': [bp for bp in df['bp']],
                    'gfr': [GFR(sc, gender, age) for sc, gender, age in zip(df['sc'], gender_column, df['age'])],
                    'gender': [sex for sex in gender_column],
                    'classification': df['Prediction']
                }
                
                csvDataFrame = pd.DataFrame(resultsForms)
                st.write(csvDataFrame)
                
                all_reports = ""

                # Generate reports for each patient
                if st.button('Generate report', key='next2'):
                    for i in range(len(resultsForms['age'])):
                        resultForm = {
                            'age': resultsForms['age'][i],
                            'bp': resultsForms['bp'][i],
                            'gfr': resultsForms['gfr'][i],
                            'gender': resultsForms['gender'][i],
                            'classification': resultsForms['classification'][i]
                        }
                        report = POST(resultForm)
                        all_reports += f"Patient {i + 1} Report:\n{report}\n\n"

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f'doctors_report_{timestamp}.txt'
                    # Save the report to a text file
                    with open(filename, 'w') as file:
                        file.write(all_reports)  # Write all reports to the file

                    # Allow report download
                    if all_reports:
                        st.download_button(
                            label="Download Report",
                            data=all_reports,
                            file_name=filename,
                            mime='text/plain'
                        )
                        st.success(f"Report has been saved to '{filename}'.")
                    else:
                        st.write('No report generated')
                
        