import os
import streamlit as st
import sqlite3
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback
from langchain.docstore.document import Document
import hashlib
import threading
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import faiss
import spacy
import pickle
import logging  # Import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
# load_dotenv()

# Initialize SpaCy for NER
logging.info("Loading SpaCy model for NER...")
nlp = spacy.load("en_core_web_sm")

# Directory for FAISS index files
FAISS_INDEX_DIR = 'faiss_index'
# Ensure the directory exists
if not os.path.exists(FAISS_INDEX_DIR):
    logging.info(f"Creating directory for FAISS index files at '{FAISS_INDEX_DIR}'")
    os.makedirs(FAISS_INDEX_DIR)

# Function to create database tables
def create_tables(conn):
    logging.info("Creating database tables if they don't exist...")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS vector_store
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  store_name TEXT,
                  vector BLOB)''')
    c.execute('''CREATE TABLE IF NOT EXISTS document_metadata
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  document_name TEXT,
                  document_hash TEXT)''')
    conn.commit()

# Function to compute MD5 hash of file content
def compute_hash(file_content):
    logging.info("Computing MD5 hash of the file content...")
    return hashlib.md5(file_content).hexdigest()

# Function to check if document exists in database
def document_exists(conn, document_name, document_hash):
    logging.info(f"Checking if document '{document_name}' exists in the database...")
    c = conn.cursor()
    c.execute('SELECT id FROM document_metadata WHERE document_name=? AND document_hash=?', (document_name, document_hash))
    return c.fetchone() is not None

# Function to save document metadata to database
def save_document_metadata(conn, document_name, document_hash):
    logging.info(f"Saving document metadata for '{document_name}'...")
    c = conn.cursor()
    c.execute('INSERT INTO document_metadata (document_name, document_hash) VALUES (?, ?)', (document_name, document_hash))
    conn.commit()

# Function to create vector store in database
def create_vector_store(chunks, store_name):
    try:
        logging.info(f"Creating vector store for '{store_name}'...")
        conn = sqlite3.connect('my_database.db')
        create_tables(conn)
        
        embeddings = OpenAIEmbeddings()
        vectors = embeddings.embed_documents(chunks)
        
        # Initialize FAISS index
        dimension = len(vectors[0])
        index = faiss.IndexFlatL2(dimension)
        
        for vec in vectors:
            vec_np = np.array(vec, dtype=np.float32)
            index.add(vec_np.reshape(1, -1))
        
        # Save FAISS index to file
        index_path = os.path.join(FAISS_INDEX_DIR, f'index.faiss')
        logging.info(f"Saving FAISS index to '{index_path}'")
        faiss.write_index(index, index_path)

        # Save embeddings to a pickle file
        with open(os.path.join(FAISS_INDEX_DIR, f'index.pkl'), 'wb') as f:
            logging.info("Saving embeddings to pickle file...")
            pickle.dump(vectors, f)
        
        conn.commit()
        conn.close()
        
        st.success(f"Vector store created and saved as {index_path} and {store_name}_index.pkl")

    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        st.error(f"Error creating vector store: {e}")

# Function to load vectors from FAISS index
def load_vector_store(store_name):
    try:
        logging.info(f"Loading vector store for '{store_name}'...")
        index_file = os.path.join(FAISS_INDEX_DIR, f'index.faiss')
        pickle_file = os.path.join(FAISS_INDEX_DIR, f'index.pkl')
        
        if not os.path.exists(index_file) or not os.path.exists(pickle_file):
            st.error(f"FAISS index file '{index_file}' or pickle file '{pickle_file}' does not exist.")
            return None, None
        
        # Load FAISS index
        index = faiss.read_index(index_file)
        
        # Load embeddings from pickle file
        with open(pickle_file, 'rb') as f:
            vectors = pickle.load(f)

        return index, vectors
    
    except Exception as e:
        logging.error(f"Error loading FAISS index: {e}")
        st.error(f"Error loading FAISS index: {e}")
        return None, None

# Function to summarize text
def summarize_text(text, language):
    logging.info(f"Summarizing text in {language}...")
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    if language == "English":
        prompt = f"Please summarize the following text in English: {text}"
    elif language == "Spanish":
        prompt = f"Por favor, resume el siguiente texto en español: {text}"
    response = llm.run(prompt)
    return response

# Function to extract entities using SpaCy
def extract_entities(text):
    logging.info("Extracting entities from text...")
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

# Function to handle PDF processing and text summarization
def process_pdfs(pdf_files, language):
    all_chunks = []
    conn = sqlite3.connect('my_database.db')
    create_tables(conn)
    logging.info(f"Processing {len(pdf_files)} PDF files...")
    
    for pdf in pdf_files:
        logging.info(f"Processing file: {pdf.name}")
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len)

        chunks = text_splitter.split_text(text=text)
        logging.info(f"Split text into {len(chunks)} chunks")

        summarized_chunks = [summarize_text(chunk, language) if len(chunk.split()) > 500 else chunk for chunk in chunks]
        store_name = pdf.name[:-4]
        file_content = pdf.getvalue()
        document_hash = compute_hash(file_content)

        if document_exists(conn, pdf.name, document_hash):
            logging.info(f"Document '{pdf.name}' exists in the database. Loading vector store...")
            index, vectors = load_vector_store(store_name)
        else:
            logging.info(f"Document '{pdf.name}' does not exist in the database. Creating new vector store...")
            create_vector_store(summarized_chunks, store_name)
            save_document_metadata(conn, pdf.name, document_hash)
            index, vectors = load_vector_store(store_name)

        all_chunks.extend(summarized_chunks)

    conn.close()
    return all_chunks, index, vectors  # Return both all_chunks and FAISS index

# Function to get the LLM based on user selection
def get_llm(model_choice):
    logging.info(f"Getting LLM pipeline for model: {model_choice}")

    if model_choice in ["BERT", "RoBERTa", "DistilBERT", "ALBERT"]:
        model_name = {
            "BERT": "bert-base-uncased",
            "RoBERTa": "roberta-base",
            "DistilBERT": "distilbert-base-uncased",
            "ALBERT": "albert-base-v2"
        }[model_choice]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        return pipeline("question-answering", model=model, tokenizer=tokenizer)
    else:
        raise ValueError("Invalid model choice")

# Function to get response from transformer-based models
def get_bert_response(pipeline, context, question):
    logging.info("Getting response from BERT model...")
    result = pipeline(question=question, context=context)
    return result['answer']

# Main function to run the Streamlit app
def main():
    logging.info("Starting Streamlit app...")
    st.set_page_config(page_title="LLM Chat App", page_icon=":robot_face:", layout="wide")
    
    # Header
    st.title("AI Receptionist")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
        ## Sidebar
        Upload your PDF files and ask questions about Location.
    """)
    st.sidebar.markdown("""
        Made with ❤️ by [DanzeeTech](https://www.danzeetech.com/)
    """)
    st.sidebar.markdown("---")
    
    # Language selection
    language = st.selectbox("Select Language", ["English", "Spanish"])
    
    # Model selection
    model_choice = st.selectbox("Select Model", ["BERT", "RoBERTa", "DistilBERT", "ALBERT","GPT"])

    # Main content area
    with st.container():
        pdf_files = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)
        
        if pdf_files:
            # Define a spinner to show while processing
            with st.spinner('Processing PDFs...'):
                logging.info("Starting PDF processing...")
                # Use threading to run the PDF processing in the background
                thread = threading.Thread(target=process_pdfs, args=(pdf_files, language))
                thread.start()

                # Wait for the thread to complete
                thread.join()

                # Once processing is done, display the input box for questions
                query = st.text_input(f"How can I help you ({language}):")
                if query:
                    all_chunks, index, vectors = process_pdfs(pdf_files, language)  # Get both chunks and index

                    context = " ".join(all_chunks)  # Combine all chunks into a single context

                    if model_choice in ["BERT", "RoBERTa", "DistilBERT", "ALBERT"]:
                        bert_pipeline = get_llm(model_choice)
                        response = get_bert_response(bert_pipeline, context, query)
                        st.markdown("---")
                        st.subheader("Answer:")
                        st.write(response)

if __name__ == '__main__':
    main()
