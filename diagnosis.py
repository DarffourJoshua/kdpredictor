from sklearn.preprocessing import  LabelEncoder
import pickle

# Load the trained model
with open('voting_model.pkl', 'rb') as file:  
    model = pickle.load(file)

with open('encoder.pkl', 'rb') as file:
    enc = pickle.load(file)

with open('normalization_model.pkl', 'rb') as file:
    norm = pickle.load(file)
    

def predictor(data):
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
        return "Positive"
    else:
        return "Negative"
        
export predictor