import os
import google.protobuf
import streamlit as st # Assuming you're using Streamlit for UI
import sqlite3
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.llms import OpenAI
from langchain.docstore.document import Document
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

# Define your Elasticsearch credentials and endpoint
ES_HOST = '776cacb5419243f1b516f5990ce3ca8b.us-central1.gcp.cloud.es.io'
ES_PORT = 9243  # Use port 9243 for Elasticsearch Cloud
# Initialize Elasticsearch client with API key
API_KEY = "THRJTV9wQUJWaGdoUXRDd0MzQ1M6ZlRLQW1xbDFRdmE1Tnh0THFGVjRlZw=="

# Initialize Elasticsearch client with API key
es = Elasticsearch(
    [f'https://{ES_HOST}:{ES_PORT}'],
    api_key=API_KEY,
    verify_certs=True
)

# Load environment variables
load_dotenv()   
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

logging.info("Logging to file and console initialized.")

# Initialize SpaCy for NER
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
                  vector BLOB)
              ''')
    c.execute('''CREATE TABLE IF NOT EXISTS document_metadata
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  document_name TEXT,
                  document_hash TEXT)
              ''')
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

# Function to get the LLM based on user selection
def get_llm(model_choice):
    #logging.info(f"Getting LLM pipeline for model: {model_choice}")
    if model_choice == "GPT":
        return OpenAI(temperature=0)
    elif model_choice in ["BERT", "RoBERTa", "DistilBERT", "ALBERT"]:
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
    
# Function to create vector store
def create_vector_store(chunks, store_name, vector_store_type='faiss'):
    try:
        logging.info(f"Creating vector store for '{store_name}'...")
        conn = sqlite3.connect('my_database.db')
        create_tables(conn)
        
        # Initialize the model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        vectors = model.encode(chunks)
        vector_dimension = vectors.shape[1]
        # Initialize FAISS index with the correct dimension
        dimension = 384  # This should match the output dimension of your model
        index = faiss.IndexFlatL2(dimension)
        logging.info(f"Vector dimension for '{store_name}': {vector_dimension}")
        
        if vector_store_type == 'faiss':
            #dimension = len(vectors[0])
            #index = faiss.IndexFlatL2(dimension)
            # Initialize FAISS index with correct dimension
            index = faiss.IndexFlatL2(vector_dimension)
            for vec in vectors:
                vec_np = np.array(vec, dtype=np.float32)
                index.add(vec_np.reshape(1, -1))
            # Save FAISS index to file
            index_path = os.path.join(FAISS_INDEX_DIR, f'{store_name}.faiss')
            logging.info(f"Saving FAISS index to '{index_path}'")
            faiss.write_index(index, index_path)
            # Save embeddings to a pickle file
            pickle_path = os.path.join(FAISS_INDEX_DIR, f'{store_name}.pkl')
            with open(pickle_path, 'wb') as f:
                logging.info("Saving embeddings to pickle file...")
                pickle.dump(vectors, f)
        # Add other vector store types if necessary...
        elif vector_store_type == 'elasticsearch':
            # Initialize Elasticsearch client with API key
            es = Elasticsearch(
                [f'https://{ES_HOST}:{ES_PORT}'],
                api_key=API_KEY,
                verify_certs=True
            )

            # Create index with a mapping for vectors
            index_body = {
                "mappings": {
                    "properties": {
                        "vector": {
                            "type": "dense_vector",
                            "dims": len(vectors[0])
                        }
                    }
                }
            }

            if not es.indices.exists(index=store_name):
                es.indices.create(index=store_name, body=index_body)

            # Index vectors into Elasticsearch
            for i, vec in enumerate(vectors):
                doc = {
                    'vector': vec.tolist()  # Convert numpy array to list
                }
                es.index(index=store_name, id=i, body=doc)

        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")

        st.success(f"Vector store created and saved as {store_name}")

    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        st.error(f"Error creating vector store: {e}")

        # index_file = os.path.join(FAISS_INDEX_DIR, f'index.faiss')
        # pickle_file = os.path.join(FAISS_INDEX_DIR, f'index.pkl')
# Function to load vector store     
def load_vector_store(store_name, vector_store_type='faiss'):
    try:
        logging.info(f"Loading vector store for '{store_name}' with type '{vector_store_type}'...")        
        if vector_store_type == 'faiss':
            index_file = os.path.join(FAISS_INDEX_DIR, f'index.faiss')
            pickle_file = os.path.join(FAISS_INDEX_DIR, f'index.pkl')
            
            if not os.path.exists(index_file) or not os.path.exists(pickle_file):
                logging.error(f"FAISS index file '{index_file}' or pickle file '{pickle_file}' does not exist.")
                st.error(f"FAISS index file '{index_file}' or pickle file '{pickle_file}' does not exist.")
                return None, None
            
            # Load FAISS index
            index = faiss.read_index(index_file)
            index_dimension = index.d
            logging.info(f"Loaded FAISS index dimension: {index_dimension}")
            
            # Load embeddings from pickle file
            with open(pickle_file, 'rb') as f:
                vectors = pickle.load(f)
            vectors = np.array(vectors, dtype=np.float32)  # Convert list to numpy array (Change 1)
            vector_dimension = vectors.shape[1]
            logging.info(f"Loaded vector dimension from pickle: {vector_dimension}")
            
            if index_dimension != vector_dimension:
                raise ValueError(f"Dimension mismatch: Index dimension {index_dimension} != Vector dimension {vector_dimension}")            
            return index, vectors
        
        # Add other vector store types if necessary...
        elif vector_store_type == 'elasticsearch':
            # Initialize Elasticsearch client
            es = Elasticsearch(
                [f'https://{ES_HOST}:{ES_PORT}'],
                api_key=API_KEY,
                verify_certs=True
            )
 
            # Check if the index exists
            index_exists = es.indices.exists(index=store_name)
            if not index_exists:
                logging.error(f"Elasticsearch index '{store_name}' does not exist.")
                st.error(f"Elasticsearch index '{store_name}' does not exist.")
                return None, None
 
            # Retrieve documents from Elasticsearch
            # Note: This assumes that you have stored the vectors in Elasticsearch as 'vector' fields
            search_result = es.search(index=store_name, body={"query": {"match_all": {}}})
            
            if not search_result['hits']['hits']:
                logging.error(f"No documents found in Elasticsearch index '{store_name}'.")
                st.error(f"No documents found in Elasticsearch index '{store_name}'.")
                return None, None
            
            vectors = []
                       
            for hit in search_result['hits']['hits']:
                vectors.append(hit['_source']['vector'])            

            # Convert vectors from lists to numpy arrays
            vectors = np.array(vectors, dtype=np.float32)  # Convert list to numpy array (Change 2)

            # Since Elasticsearch does not have an index object like FAISS, return None
            #logging.info(f"Loading vector store for '{store_name}' with type '{vector_store_type}'\n index is '{index}' \n vectors are: '{vectors}' ") 
            return None, vectors  # Return None for index, vectors for Elasticsearch

        else:
            logging.error(f"Unsupported vector store type: {vector_store_type}")
            st.error(f"Unsupported vector store type: {vector_store_type}")
            return None, None
             
    except Exception as e:
        logging.error(f"Error loading vector store: {e}")
        st.error(f"Error loading vector store: {e}")
        return None, None

# Function to summarize text using BART
def summarize_text_transformer(text, language):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=90, min_length=30, do_sample=False)
    #logging.info(f"Actual Text: {text}")
    #logging.info(f"Summarizing text: {summary}")
    return summary[0]['summary_text']

# Function to save summarized chunks to PDF
def save_summarized_chunks_to_pdf(summarized_chunks, output_pdf_path):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Summarized Content', 0, 1, 'C')

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for chunk in summarized_chunks:
        pdf.multi_cell(0, 10, chunk)
    pdf.output(output_pdf_path)
    logging.info(f"Summarized content saved to {output_pdf_path}")

# Function to process PDF files
def process_pdfs(pdf_files, language, vector_store_type):
    conn = sqlite3.connect('my_database.db')
    create_tables(conn)
    
    all_chunks = []
    all_vectors = []
    all_indices = []

    for pdf in pdf_files:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        logging.info("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        chunks = text_splitter.split_text(text)

        summarized_chunks = []
        for chunk in chunks:
            summarized_chunk = summarize_text_transformer(chunk, language)
            summarized_chunks.append(summarized_chunk)
        
        logging.info(f"Summarized {len(summarized_chunks)} chunks")

        store_name = pdf.name[:-4]
        file_content = pdf.read()  # Corrected from pdf.getvalue()
        document_hash = compute_hash(file_content)
        #logging.info(f"PDF file content: {file_content}")
        logging.info(f"Document Hash: {document_hash}")

        if document_exists(conn, pdf.name, document_hash):
            logging.info(f"Document exists in the database")
            index, vectors = load_vector_store(store_name, vector_store_type)
            logging.info(f"Loaded vector store for '{store_name}'")
        else:
            logging.info(f"Document doesn't exist in the database, creating a new one")
            create_vector_store(summarized_chunks, store_name, vector_store_type)
            save_document_metadata(conn, pdf.name, document_hash)
            index, vectors = load_vector_store(store_name, vector_store_type)
            logging.info(f"Created and loaded vector store for '{store_name}'")
        
        # Log the results of load/create vector store
        if index is None or vectors is None:
            logging.error(f"Failed to load/create vector store for '{store_name}'")

        all_chunks.extend(summarized_chunks)

        if vectors is not None and vectors.size > 0:  # Check if vectors array is not empty
            all_vectors.extend(vectors)
        all_indices.append(index)
    
    conn.close()
    return all_chunks, all_indices, all_vectors

# Measure performance function with improved error handling
def measure_performance(all_chunks, indices, vectors, query, vector_store_type):
    start_time = time.time()
    context = " ".join(all_chunks)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    # Encode the query using the same model used for creating vectors
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = model.encode([query])[0]
    query_vector_dimension = query_vector.shape[0]
    logging.info(f"Query vector dimension: {query_vector_dimension}")

    results = []
    try:
        if vector_store_type == 'faiss':
            query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            if indices:
                for index in indices:
                    if index.d != query_vector.shape[1]:
                        raise ValueError(f"Dimension mismatch: Index dimension {index.d} != Query vector dimension {query_vector.shape[1]}")
                    dists, ids = index.search(query_vector, 5)
                    results.extend([context[i] for i in ids[0]])
            else:
                raise ValueError("FAISS index is empty or not loaded properly.")
            logging.info(f"The Question result: {results}")

        elif vector_store_type == 'elasticsearch':
            body = {
                "query": {
                    "knn": {
                        "field": "vector",
                        "query_vector": query_vector.tolist(),
                        "k": 5
                    }
                }
            }
            search_results = es.search(index="vector_store", body=body)
            results.extend(hit['_source'] for hit in search_results['hits']['hits'])

        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")

    except Exception as e:
        logging.error(f"Error in measure_performance: {e}")
        st.error(f"Error in measure_performance: {e}")
        return [], 0

    elapsed_time = time.time() - start_time

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
    model_choice = st.selectbox("Select Model", ["BERT", "RoBERTa", "DistilBERT", "ALBERT", "GPT"])
    vector_store_type = st.selectbox("Select Vector Store", ["faiss", "elasticsearch"])

    with st.container():
        pdf_files = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)
        if pdf_files:
            with st.spinner('Processing PDFs...'):
                all_chunks, indices, vectors = process_pdfs(pdf_files, language, vector_store_type)

            #................................
            if model_choice == "GPT":
                llm = get_llm(model_choice)
                chain = load_qa_chain(llm=llm, chain_type="map_reduce")

                with get_openai_callback() as cb:
                    if language == "Spanish":
                        query = f"Por favor, responde en español: {query}"

                        response = chain.run(input_documents=[Document(page_content=chunk) for chunk in all_chunks], question=query)
                        st.markdown("---")
                        st.subheader("Answer:")
                        st.write(response)
                        st.markdown("---")
                        st.write(f"Total Tokens: {cb.total_tokens}")
                        st.write(f"Prompt Tokens: {cb.prompt_tokens}")
                        st.write(f"Completion Tokens: {cb.completion_tokens}")
                        st.write(f"Total Cost (USD): ${cb.total_cost:.5f}")            
            #................................
            if np.size(vectors) > 0:  # Correct way to check if vectors array is not empty
                query = st.text_input(f"How can I help you ({language}):")
                logging.info(f"The user Question: {query}")
                if query:
                    results, elapsed_time = measure_performance(all_chunks, indices, vectors, query, vector_store_type)
                    
                    st.markdown("---")
                    st.subheader("Answer:")
                    st.write(results)
                    st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")
            else:
                logging.info("No vectors found.")
           
 
if __name__ == '__main__':
    main()


