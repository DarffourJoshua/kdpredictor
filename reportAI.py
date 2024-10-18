from dotenv import load_dotenv
import os
import bs4
# from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

# import requests

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
vectorStore = Chroma.from_documents(
    collection_name="example_collection",
    documents=all_splits, 
    embedding=CohereEmbeddings(model='embed-english-v3.0'),
    persist_directory="./chroma_langchain_db",
)

retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



question = "What is the patient's condition, and what are the potential treatment options?"



def POST(data):
    classification, GFR, bp, age, gender = data.values()
    try:
        # for row in data:
        #     age = row['age']
        #     bp = row['bp']
        #     gfr = row['gfr']
        #     gender = row['gender']
        #     classification = row['classification']
        
        # Define the prompt message as a doctor creating a report based on the model's prediction and user inputs
        system_prompt = f"""
            You are a nephrologist. Based on the following patient data and model predictions, write a concise medical report for the patient:
                
            classification:{classification}, 
            GFR:{GFR}, 
            bp:{bp}, 
            age:{age}, 
            gender: {gender}
                    
            Question: {question}
            The report should include an assessment of the patient's condition, potential treatment options, and recommended follow-up actions.
            Your Report:
        """

        prompt = PromptTemplate.from_template(system_prompt)
        # req_data = await request.json()
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            |   prompt
            |   llm
            |   StrOutputParser()
        )
        response = rag_chain.invoke(question)
        print(response)
        return response
    except Exception as e:
        print(str(e))
        return ( str(e))