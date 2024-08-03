import os
import google.protobuf
import streamlit as st  # Assuming you're using Streamlit for UI
import sqlite3
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import faiss
import spacy
import pickle
import logging
import time
from fpdf import FPDF
import hashlib
import threading
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv()
StoreName=[]
# Set up logging
log_filename = 'app.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),  # Save logs to file
        logging.StreamHandler()  # Also output logs to console
    ]
)

# Function to process PDF files
def process_pdfs(pdf_files, language, vector_store_type):
    logging.info("I am in process_pdfs")
    
    all_chunks = []
    all_vectors = []
    all_indices = []

    store_name = 'pdfsara'
    all_chunks = 'Hafeez, Overnight stay includes breakfast in one of our connecting rooms or suites. From Buckingham Palace to the West End, you’ll be perfectly positioned to explore all the capital’s must-see sights.'
    all_indices = [None]
    all_vectors = [
        1.32649943e-01, -1.14231091e-02,  3.56042273e-02,  7.99739063e-02,
        6.62042052e-02,  1.74281431e-05, -1.35741634e-02, -9.53154340e-02,
        2.80577131e-02, -1.13605084e-02, -7.43318498e-02,  5.39228246e-02,
       -4.26384388e-03, -2.17804760e-02,  4.44805399e-02, -4.52483399e-03,
        7.06844777e-02, -2.35754047e-02,  4.40998636e-02, -2.56907791e-02,
       -3.57827405e-03, -5.33451997e-02,  4.57250588e-02,  3.87085937e-02,
       -9.89709049e-03, -1.88702177e-02,  4.76171933e-02,  2.50977352e-02,
       -2.78116614e-02, -6.91720471e-02, -1.72279701e-02,  2.90451143e-02,
       -4.89863940e-02, -2.29480653e-03,  1.33082159e-02,  4.33021523e-02,
        4.27962840e-02, -2.59694718e-02,  3.48613672e-02, -4.74782698e-02]

    return store_name, all_chunks, all_indices, all_vectors

# Measure performance function with improved error handling
def measure_performance(store_name, all_chunks, indices, vectors, query, vector_store_type):
    logging.info("I am in measure_performance funciton")
    start_time = time.time()
    context = " ".join(all_chunks)

    elapsed_time = time.time() - start_time
    results = '{results}'
    elapsed_time = '{elapsed_time}'
    logging.info(f"results '{results}' and elapsed time '{elapsed_time}'")
    return results, elapsed_time

# Main function
def main():
    st.set_page_config(page_title="LLM Chat App", page_icon=":robot_face:", layout="wide")

    st.title("AI Receptionist")
    st.markdown("---")

    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
        ## Sidebar
        Upload your PDF files and ask questions about Location.
    """)
    st.sidebar.markdown("""
        Made with ❤️ by [DanzeeTech](https://www.danzeetech.com/)
    """)
    st.sidebar.markdown("---")

    language = st.selectbox("Select Language", ["English", "Spanish"])
    model_choice = st.selectbox("Select Model", ["BERT", "RoBERTa", "DistilBERT", "ALBERT"])
    vector_store_type = st.selectbox("Select Vector Store", ["faiss", "elasticsearch"])

    pdf_files = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)

    if pdf_files and 'store_name' not in st.session_state:
        with st.spinner('Processing PDFs...'):
            logging.info(f"I am in main, and going to call process_pdfs function with \n'{pdf_files}', '{language}', \n'{vector_store_type}'")
            store_name, all_chunks, indices, vectors = process_pdfs(pdf_files, language, vector_store_type)
            
            st.session_state['store_name'] = store_name
            st.session_state['all_chunks'] = all_chunks
            st.session_state['indices'] = indices
            st.session_state['vectors'] = vectors
            StoreName=store_name
            logging.info(f" store_name '{store_name}' and StoreName '{StoreName}' ")
            logging.info(f"Processed PDF and stored results in session_state")

    if 'vectors' in st.session_state and np.size(st.session_state['vectors']) > 0:
        query = st.text_input(f"How can I help you ({language}):")
        logging.info(f"The user Question: {query}")

        if query:
            logging.info("I am going to call measure_performance function")
            results, elapsed_time = measure_performance(st.session_state['store_name'], st.session_state['all_chunks'], st.session_state['indices'], st.session_state['vectors'], query, vector_store_type)
            
            st.markdown("---")
            st.subheader("Answer:")
            st.write(results)
            # st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")
    else:
        logging.info("No vectors found.")

if __name__ == '__main__':
    main()
