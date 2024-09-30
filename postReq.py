import requests
import json
import streamlit as st

def post(data):
            try:
                res = requests.post(
                    './reportAI.py', 
                    data=json.dumps(data), 
                    headers={'Content-Type':'application/json'}
                )
                
                if res.status_code == 200:
                    with open('doctors_report', 'rb1') as file:
                        json.dump(res.json(), file, indent=4)
                    
                    st.success('Report generated successfully')
                    st.json(res.json())
                
                else:
                    st.error(f'Failed to genetate report. Please try again later')
            
            except Exception as e:
                st.error(f'Something, went wrong: {str(e)}')
                