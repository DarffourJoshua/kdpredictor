

def upload_file(file):
    #Check if the file is uploaded
    if file is not None:
        df = pd.read_csv(file)
        
        # Store the id column temporarily
        if 'id' in df.columns:
            id_column = df['id']
        
        # Drop the id and classification columns
        if 'id' in df.columns and 'classification' in df.columns:
            df.drop(["id", "classification"], axis=1, inplace=True)
            
        for i in range(df.shape[0]):
            if df.iloc[i,19] = '\tyes':
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
    
export upload_file

# import numpy as np
# import pandas as pd
# import streamlit as st 
# from sklearn.preprocessing import  LabelEncoder
# import pickle

# # Load the trained model
# with open('dtc_model.pkl', 'rb') as file:  
#     model = pickle.load(file)

# with open('encoder.pkl', 'rb') as file:
#     enc = pickle.load(file)


# def main():
#     st.header('Single File Upload')
#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv", accept_multiple_files=False)
    
#     #Check if the file is uploaded
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
        
#          # Store the id column temporarily
#         if 'id' in df.columns:
#             id_column = df['id']
        
#         # Drop the id and classification columns
#         if 'id' in df.columns and 'classification' in df.columns:
#             df.drop(["id", "classification"], axis=1, inplace=True)
            
#         for i in range(df.shape[0]):
#             if df.iloc[i,19] in [' yes','\tyes']:
#                 df.iloc[i,19]='yes'
#             if df.iloc[i,19]=='\tno':
#                 df.iloc[i,19]='no'
#             if df.iloc[i,20]=='\tno':
#                 df.iloc[i,20]='no'
#             if df.iloc[i,15]=='\t?':
#                 df.iloc[i,15]=np.nan
#             if df.iloc[i,15]=='\t43':
#                 df.iloc[i,15]='43'
#             if df.iloc[i,16]=='\t?':
#                 df.iloc[i,16]=np.nan
#             if df.iloc[i,16]=='\t6200':
#                 df.iloc[i,16]= '6200'
#             if df.iloc[i,16]=='\t8400':
#                 df.iloc[i,16]= '8400'
#             if df.iloc[i,17]=='\t?':
#                 df.iloc[i,17]=np.nan

#         #Seperate the categorical columns from the numerical columns
#         cat_cols = [col for col in df.columns if df[col].dtype == "object"]
#         num_cols = [col for col in df.columns if df[col].dtype != "object"]

#         # filling null values, we will use two methods, random sampling for higher null values and 
#         def random_value_imputation(feature):
#             random_sample = df[feature].dropna().sample(df[feature].isna().sum())
#             random_sample.index = df[df[feature].isnull()].index
#             df.loc[df[feature].isnull(), feature] = random_sample
            
#         # mean/mode sampling for lower null values
#         def impute_mode(feature):
#             mode = df[feature].mode()[0]
#             df[feature] = df[feature].fillna(mode)

#         # filling num_cols null values using random sampling method
#         for col in num_cols:
#             random_value_imputation(col)

#         # filling cat_cols null values using mode sampling method
#         for col in cat_cols:
#             impute_mode(col)
        
#         if st.button('Predict', type="secondary"):
        
#             for cat in enc:
#                 if cat in df.columns:
#                     le = LabelEncoder()
#                     le.classes_ = np.array(enc[cat] + ['Unknown'])
            
#                     # Strip leading whitespace
#                     df[cat] = df[cat].str.lstrip()
            
#                     # Handle unknown categories by setting them to 'Unknown'
#                     df[cat] = df[cat].apply(lambda x: x if x in le.classes_ else 'Unknown')
            
#                     # Transform the column
#                     df[cat] = le.transform(df[cat])
                    
#             #read the columns from the dataframe to the model and create a new column for the prediction
#             predictions = model.predict(df.values)
#             df['Prediction'] = predictions
#             df['Prediction'] = df['Prediction'].map({1: 'yes', 0: 'no'})
            
#             # Add the id column back to the DataFrame
#             df.insert(0, 'id', id_column)
            
#         if st.button('Save'):
#             df.to_csv('prediction.csv', index=False)

        

#         st.write(df)
    
# if __name__ == '__main__':
#     main()
