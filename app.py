import streamlit as st
import pandas as pd
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# documents
documents = []

# Load your embeddings and vector store
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

# Load your vector store
persist_directory = '../data/chroma/'
vectordb = Chroma.from_documents(persist_directory)

# Load your LLM
llm = Ollama(model="mistral")

# Build prompt
template = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use five sentences maximum. Keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 

{context}

Question: {question}

Helpful Answer:
"""
prompt = PromptTemplate.from_template(template)

# Build QA chain
chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Streamlit app
st.title("Knitting FAQ")
question = st.text_input("Ask your question here:")

if st.button("Get Answer"):
    result = chain({"query": question})
    answer = result["result"]
    st.write("Answer:")
    st.write(answer)
