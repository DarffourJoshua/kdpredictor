from sklearn.preprocessing import  LabelEncoder
import pandas as pd
import numpy as np

def predictor(data, enc, norm, model):
    # Define the column names
    cols = ["age", "bp", "sg", "al", "su", "sc", "sod", "hemo", "pcv", "rc", "htn", "dm"]
    
    df=pd.DataFrame([list(data.values())], columns= cols)
        
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
   
    for cat in enc:
        if cat in df.columns:
            le = LabelEncoder()
            le.classes_ = np.array(enc[cat] + ['Unknown'])
        
            if df[cat].dtype == 'object':
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
        return "Positive"
    else:
        return "Negative"