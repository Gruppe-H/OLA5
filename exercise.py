
import streamlit as st
# %%
"""
# Exercise 5
### By Gruppe H (Caroline og Maria)
"""

# %%
#!pip install streamlit

# %%
#!streamlit hello

# %%
#!pip install langchain

# %%
#!pip install langdetect

# %%
#!pip install -U torch

# %%
"""
## Set up Enviroment 
"""

# %%
import os
import pandas as pd

# %%
import langdetect
from langdetect import DetectorFactory, detect, detect_langs

# %%
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# %%

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# %%
# Embedding facilities
from langchain.embeddings import HuggingFaceEmbeddings

# %%
# Pipelines
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

# %%
"""
## Load Documents
Our first task is to collect and load documents from various sources and formats in the context of our chosen domain, which is **knitting for beginners**.

We have chosen to load content from pdf files, YouTube and Wikipedia.
"""

# %%
import myloadlib
from myloadlib import loadDir, loadFile, loadWiki, loadYoutube, readAPI

# %%
import importlib 
importlib.reload(myloadlib)

# %%
# Collect all here
documents = []

# %%
"""
### Load single pdf files

#### File 1
"""

# %%
file = "./data/knitting_pdfs/knit.pdf"

# %%
#!pip install pypdf

# %%
docs = myloadlib.loadFile(file)

# %%
documents.extend(docs)
len(documents)

# %%
# metadata of loaded Document
docs[0].metadata 

# %%
"""
Content of page [0] = page 1. 
"""

# %%
documents[0].page_content
#docs[0].page_content[:1000]
# First 1000 charactors.

# %%
"""
#### File 2
"""

# %%
file2 = "./data/knitting_pdfs/knitting-handbook.pdf"

# %%
docs = myloadlib.loadFile(file2)

# %%
documents.extend(docs)
len(documents)

# %%
docs[1].metadata 

# %%
documents[1].page_content

# %%
"""
### Load YouTube
"""

# %%
url = 'https://www.youtube.com/watch?v=Zjq0MoUZqVY'
save_dir="./youtube/"

# %%
url

# %%
lang = 'en'

# %%
#!pip install youtube-transcript-api

# %%
#!pip install pytube

# %%
docs = myloadlib.loadYoutube(url, lang)

# %%
documents.extend(docs)
len(documents)

# %%
documents[61].type

# %%
documents[61].page_content

# %%
"""
### Load wikipedia page
"""

# %%
subject = "Knitting"

# %%
lang = 'en'

# %%
#!pip install wikipedia

# %%
docs = myloadlib.loadWiki(subject, lang, 2)

# %%
documents.extend(docs)

# %%
"""
Should be 4 at the moment, but will update everytime its run, and or other Docs/documents are ran again
"""

# %%
len(documents)

# %%
"""
## Chunking
Now we will be chunking our documents, which means breaking down our texts into smaller, more manageable chunks to prepare it for AI processing.
"""

# %%
#!pip install spacy
#!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.0.0/en_core_web_md-3.0.0.tar.gz

# %%
#!pip install wordcloud

# %%
import myutils2
from myutils2 import chunkDocs, langDetect, wordCloud

# %%
import importlib 
importlib.reload(myutils2)

# %%
splits = myutils2.chunkDocs(documents, 350)  
splits

# %%
len(splits)

# %%
splits[70]

# %%
df = pd.DataFrame(splits, columns=['page_content', 'metadata', 'type'])
df.sample(3)

# %%
df['page_content'][0]

# %%
df['metadata'][0]

# %%
#!pip install scapy

# %%
#!bash
#!python3 -m spacy download en_core_web_md

# %%
"""
### Data Visualization
To visually represent the data of our texts, we have created a word cloud. On the word cloud, we can see which words apear more frequently as they appear bigger.
"""

# %%
im, longstring = myutils2.wordCloud(df, 'page_content')

# %%
im

# %%
"""
## Embeddings
"""

# %%
model_name = "sentence-transformers/all-mpnet-base-v2"
# model_name = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

# %%
#!pip install sentence-transformers

# %%
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# %%
from langchain.vectorstores import FAISS

# %%
#!pip install faiss-cpu

# %%
db = FAISS.from_documents(splits, embeddings)

# %%
"""
## Storing the Embeddings in Vector DB
"""

# %%
#!pip install chromadb

# %%
db = Chroma.from_documents(splits, embeddings)

# %%
persist_directory = '../data/chroma/'

# Create the vector store
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)
vectordb.persist()

# %%
vectordb._collection.count()

# %%
"""
## Similarity Search
"""

# %%
query = 'what is the best yarn for beginners?'

# %%
answer = vectordb.similarity_search(query, k=3)
answer

# %%
for d in answer:
    print(d.page_content)

# %%
for d in answer:
    print(d.metadata)

# %%
"""
## Information Retrieval
"""

# %%
q1 = 'What does "K2tog" mean?'

# %%
q2 = 'What are stitch markers and how do I use them?'

# %%
q3 = 'How do I cast on stitches?'

# %%
q4 = "What's the difference between circular needles and straight needles?"

# %%
answer = vectordb.max_marginal_relevance_search(q1, k=2, fetch_k=5)
for d in answer:
    print(d.page_content)

# %%
answer = vectordb.max_marginal_relevance_search(q2, k=2, fetch_k=5)
for d in answer:
    print(d.page_content)

# %%
answer = vectordb.max_marginal_relevance_search(q3, k=2, fetch_k=5)
for d in answer:
    print(d.page_content)

# %%
answer = vectordb.max_marginal_relevance_search(q4, k=2, fetch_k=5)
for d in answer:
    print(d.page_content)

# %%
"""
## Large Language Model
"""

# %%
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# %%
#!ollama list

# %%
llm = Ollama(model="mistral", callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))

# %%
# Build prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use five sentences maximum. Keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 

{context}

Question: {question}

Helpful Answer:
"""

# %%
prompt = PromptTemplate.from_template(template)
chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt})

# %%
question1 = 'What does "K2tog" mean?'

# %%
result1 = chain({"query": question1})
result1["result"]

# %%
question2 = 'What are stitch markers and how do I use them?'

# %%
result2 = chain({"query": question2})

# %%
question3 = 'How do I cast on stitches?'

# %%
result3 = chain({"query": question3})

# %%
question4 = "What's the difference between circular needles and straight needles?"

# %%
result4 = chain({"query": question4})

# %%
question5 = 'What does water taste like?'

# %%
result5 = chain({"query": question5})

# %%
question6 = 'what color pants do i have on?'

# %%
result6 = chain({"query": question6})

