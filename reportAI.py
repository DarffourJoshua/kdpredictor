from dotenv import load_dotenv
import os
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import requests

#loading environment variables
load_dotenv()

#initialize the model
llm = ChatGroq(model="llama3-8b-8192")

#scrap ckd info from the website
bs4_strainer = bs4.SoupStrainer(attrs={
    'data-identity': ['headline', 'paragraph-element', 'unordered-list']
})

loader = WebBaseLoader(
    web_paths=("https://my.clevelandclinic.org/health/diseases/15096-chronic-kidney-disease",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

#split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

#store the chunks in a vector database structure
vectorstore = Chroma.from_documents(
    documents=all_splits, 
    embedding=CohereEmbeddings(model='embed-english-v3.0')
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def POST(request):


model_results = {
    classification: '',
    GFR: '',
    blood_pressure: '',
    age: '',
    gender: '',
}


# Define the prompt message as a doctor creating a report based on the model's prediction and user inputs
system_prompt = f"""
You are a nephrologist. Based on the following patient data and model predictions, write a concise medical report for the patient:

- Diagnosis: {model_results['classification']}
- Glomerular Filtration Rate (GFR): {model_results['GFR']}
- Blood Pressure: {model_results['blood_pressure']}
- Age: {model_results['age']}
- Gender: {model_results['gender']}

The report should include an assessment of the patient's condition, potential treatment options, and recommended follow-up actions.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", model_results),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"model_results": model_results})
