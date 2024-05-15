import streamlit as st
import pandas as pd
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import myloadlib
from myloadlib import loadFile
import myutils2
from myutils2 import chunkDocs, langDetect, wordCloud

def main():
    st.title('Interactive PDF to Text and QA Application')

    # File uploader widget to get a PDF file from the user
    st.header('Upload PDF File:')
    pdf_file = st.file_uploader('Upload a PDF file', type=['pdf'])

    # Input widget to get the question from the user
    st.header('Enter Your Question:')
    question = st.text_input('Type your question here')

    # Button to trigger the processing of the uploaded PDF and question
    if st.button('Get Answer'):
        if pdf_file is not None and question:
            # Process the uploaded PDF file to extract text content
            pdf_text = myloadlib.loadFile(pdf_file)
            if pdf_text:
                answer = get_answer(pdf_text, question)
                st.subheader('Answer:')
                st.write(answer)
            else:
                st.warning('Failed to extract text from the PDF file. Please try again.')
        else:
            st.warning('Please upload a PDF file and provide a question')


def get_answer(documents, question):
    splits = myutils2.chunkDocs(documents, 350)  
    df = pd.DataFrame(splits, columns=['page_content', 'metadata', 'type'])

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    db = Chroma.from_documents(splits, embeddings)
    persist_directory = '../data/chroma/'

    # Create the vector store
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    
    llm = Ollama(model="mistral", callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))
    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use five sentences maximum. Keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer. 

    {context}

    Question: {question}

    Helpful Answer:
    """
    
    prompt = PromptTemplate.from_template(template)
    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return chain({"query": question})


if __name__ == "__main__":
    main()
