import os
import google.protobuf
import streamlit as st # Assuming you're using Streamlit for UI
import sqlite3
import numpy as np
#import tensorflow as tf
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
st.set_page_config(page_title="LLM Chat App", page_icon=":robot_face:", layout="wide")
# Setup CUDA_VISIBLE_DEVICES environment variable if you do not have a GPU or want to force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#.....................................................
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
    logging.info("I am in creating SQLite Creating database tables if they don't exist...")
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS vector_store
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,   
                  store_name TEXT,  
                  vector BLOB)
              ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS document_metadata
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         document_name TEXT,
         document_hash TEXT,
         all_chunks BLOB,
         all_indices BLOB,
         all_vectors BLOB)
    ''')
    conn.commit()
# Initialize logging
logging.basicConfig(level=logging.INFO)
# Function to delete all data from specified tables
def delete_data_from_tables( pdfname,document_hash):
    
    try:
        conn = sqlite3.connect('my_database.db')
        create_tables(conn)
        c = conn.cursor()
        # Delete all data from the vector_store table
        c.execute('DELETE FROM vector_store')
        # Delete all data from the document_metadata table
        c.execute('DELETE FROM document_metadata')
        # Commit the changes
        conn.commit()
        logging.info("All data removed from vector_store and document_metadata tables.")
    except Exception as e:
        logging.error(f"Error deleting data from tables: {e}")
        conn.rollback()
    #conn.close
def removing_vector_elasticsearch(srote_name):
        # Index name
    index_name = srote_name
    # Check if the index exists before trying to delete it
    if es.indices.exists(index=index_name):
        # Delete the index
        response = es.indices.delete(index=index_name)
        print(f"Index '{index_name}' deleted successfully: {response}")
    else:
        print(f"Index '{index_name}' does not exist.")
# Function to compute MD5 hash of file content
def compute_hash(file_content):
    logging.info("Computing MD5 hash of the file content...")
    return hashlib.md5(file_content).hexdigest()

# Function to check if document exists in database
def document_exists(conn, document_name, document_hash):
    #logging.info(f"Checking if document '{document_name}' exists in the database...")
    c = conn.cursor()
    c.execute('SELECT id FROM document_metadata WHERE document_name=? AND document_hash=?', (document_name, document_hash))
    return c.fetchone() is not None

# Function to save document metadata to database
#def save_document_metadata(conn, document_name, document_hash):
def save_document_metadata(conn, document_name, document_hash, all_chunks, all_indices, all_vectors):
    logging.info(f"Saving document metadata for '{document_name}'...")
    
    c = conn.cursor()
    #c.execute('INSERT INTO document_metadata (document_name, document_hash) VALUES (?, ?)', (document_name, document_hash))
    c.execute('''INSERT INTO document_metadata 
                 (document_name, document_hash, all_chunks, all_indices, all_vectors) 
                 VALUES (?, ?, ?, ?, ?)''', 
              (document_name, document_hash, 
               pickle.dumps(all_chunks), 
               pickle.dumps(all_indices), 
               pickle.dumps(all_vectors)))    
    conn.commit()

def load_document_metadata(conn, document_name, document_hash):
    c = conn.cursor()
    c.execute('''SELECT all_chunks, all_indices, all_vectors 
                 FROM document_metadata 
                 WHERE document_name=? AND document_hash=?''', 
              (document_name, document_hash))
    row = c.fetchone()
    if row:
        all_chunks = pickle.loads(row[0])
        all_indices = pickle.loads(row[1])
        all_vectors = pickle.loads(row[2])
        return all_chunks, all_indices, all_vectors
    return None, None, None

def create_vector_store_type_elasticsearch(vectors, chunks, store_name):
    logging.info("I am in Create_vector_Store_type_elasticsearch")
    try:
        # Define index body
        index_body = {
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": len(vectors[0])  # Number of dimensions of the vector
                    },
                    "text": {
                        "type": "text"
                    }
                }
            }
        }
        logging.info("Checking if index exists...")
        # Check if index exists
        logging.info("Going into if block to create index...")

        if not es.indices.exists(index=store_name):
            logging.info("Index does not exist. Creating index...")           
            # Create index with the specified mappings
            response = es.indices.create(index=store_name, body=index_body)
            logging.info(f"Index creation response: {response}")

            # Index vectors and chunks into Elasticsearch
            logging.info("Indexing vectors and text chunks...")
            for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
                doc = {
                    'vector': vec.tolist(),  # Convert numpy array to list if using numpy
                    'text': chunk
                }
                es.index(index=store_name, id=i, body=doc)
            logging.info(f"Created and indexed Elasticsearch index '{store_name}'.")
        else:
            logging.info(f"Elasticsearch index '{store_name}' already exists.")
    except Exception as e:
        logging.error(f"Error creating Elastic vector store: {e}")
        st.error(f"Error creating Elastic vector store: {e}")

#Function to create vector store
def create_vector_store(document_hash, chunks, store_name, vector_store_type='faiss'):
    #create_vector_store(document_hash,summarized_chunks, store_name, vector_store_type)
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
        #logging.info(f"Vector dimension for '{store_name}': {vector_dimension}")  

        if vector_store_type == 'faiss': 
            # Initialize FAISS index with correct dimension
            index = faiss.IndexFlatL2(vector_dimension)
            for vec in vectors:
                vec_np = np.array(vec, dtype=np.float32)
                index.add(vec_np.reshape(1, -1))
            
            index.add(vectors)
            # Save FAISS index to file
            index_path = os.path.join(FAISS_INDEX_DIR, f'{store_name}.faiss')
            logging.info(f"Saving FAISS index to '{index_path}'")
            faiss.write_index(index, index_path)
            # Save embeddings to a pickle file
            pickle_path = os.path.join(FAISS_INDEX_DIR, f'{store_name}.pkl')
            with open(pickle_path, 'wb') as f:
                logging.info("Saving embeddings to pickle file...")
                pickle.dump(vectors, f)
            return index, vectors
        
            # Add other vector store types if necessary...
        elif vector_store_type == 'elasticsearch':
            logging.info(f"I am in Vstore type elastic search inside create_vector_store")
            create_vector_store_type_elasticsearch(vectors, chunks, store_name)
            return None, vectors
        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")
        
        return vectors
        st.success(f"Vector store created and saved as {store_name}")

    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        st.error(f"Error creating vector store: {e}")

        # index_file = os.path.join(FAISS_INDEX_DIR, f'index.faiss')
        # pickle_file = os.path.join(FAISS_INDEX_DIR, f'index.pkl')

def load_vector_store_type_elasticsearch(store_name):
    # Check if the index exists
    index_exists = es.indices.exists(index=store_name)
    if not index_exists:
        logging.error(f"Elasticsearch index '{store_name}' does not exist.")
        st.error(f"Elasticsearch index '{store_name}' does not exist.")
        return None, None
    else:
        logging.info(f"Elasticsearch index '{store_name}' exist.")
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
    logging.info(f"Loading vector store for '{store_name}' with type elasticsearch  \n vectors are: '{vectors}' ") 
    return vectors  # Return None for index, vectors for Elasticsearch
    
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

            vectors = load_vector_store_type_elasticsearch(store_name)

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
    summary = summarizer(text, max_length=500, min_length=60, do_sample=False)
    #logging.info(f"Actual Text: {text}")
    #logging.info(f"Summarizing text: {summary}")
    return summary[0]['summary_text']

# Function to save summarized chunks to PDF

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def save_summarized_chunks_to_pdf(summarized_chunks, output_pdf_path):
    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica", 12)
    y = height - 40  # Starting vertical position

    for chunk in summarized_chunks:
        if y < 40:  # Check if we need to add a new page
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 40
        c.drawString(40, y, chunk)
        y -= 20  # Move down for the next line

    c.save()
    logging.info(f"Summarized content saved to {output_pdf_path}")

# Function to process PDF files
def process_pdfs(pdf_files, language, vector_store_type):
    conn = sqlite3.connect('my_database.db')
    create_tables(conn)
    #logging.info(f" I am in process_pdfs with pdf_files {pdf_files}")
    #store_name = pdf.name[:-4]
    all_chunks = []
    all_vectors = []
    all_indices = []

    for pdf in pdf_files:
        store_name = pdf.name[:-4]
        file_content = pdf.read()  # Corrected from pdf.getvalue()
        document_hash = compute_hash(file_content)
        #logging.info(f"PDF file content: {file_content}")
        #logging.info(f"Document Hash: {document_hash}, store_name  '{store_name}'")

        if document_exists(conn, pdf.name, document_hash):
            logging.info(f"Document exists in the database")
            all_chunks, all_indices, all_vectors = load_document_metadata(conn, pdf.name, document_hash)
            #index, vectors = load_vector_store(store_name, vector_store_type)
            #logging.info(f"Loaded vector store for '{store_name}'") 
            
            #if all_chunks and all_indices and all_vectors:
            if all_chunks and all_vectors:
                logging.info(f"Document '{pdf.name}' found in the database.")
                continue              
        else:
            logging.info(f"Document doesn't exist in the database, creating a new one")
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            logging.info("Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            chunks = text_splitter.split_text(text)

            logging.info(f"Chunks:  '{chunks}'")
            
            summarized_chunks = []
            summarized_chunks = chunks
        
            #for chunk in chunks:
            #    summarized_chunk = summarize_text_transformer(chunk, language)
            #    summarized_chunks.append(summarized_chunk)
            
            #logging.info(f"summarised chunk: '{summarized_chunks}'")
            save_summarized_chunks_to_pdf(summarized_chunks, "summarise.pdf")
            
            
            #logging.info(f"Summarized {len(summarized_chunks)} chunks")
            #logging.info(f"PDF file content: {file_content}")
            #logging.info(f"Document Hash: {document_hash}, store_name  '{store_name}'")
           
            #vectors = create_vector_store(summarized_chunks, store_name, vector_store_type)
            index, vectors = create_vector_store(document_hash,summarized_chunks, store_name, vector_store_type)

            #logging.info(f"successfully created vector store and getback to process_pdfs with vectors")
            all_chunks.extend(summarized_chunks)
            all_vectors.extend(vectors)
            all_indices.append(index)

            #logging.info(f"document_name '{pdf.name}'\n document_hash '{document_hash}'\n  all_chunks '{all_chunks}'\n   all_indices '{all_indices}'\n all_vectors '{all_vectors}'\n ")
            #logging.info(f"Storing data into sqlite when new documents is fed ")
            save_document_metadata(conn, pdf.name, document_hash, summarized_chunks, all_indices, all_vectors)
            index, vectors = load_vector_store(store_name, vector_store_type)
            logging.info(f"Created and loaded vector store for '{store_name}'")
            
        # Log the results of load/create vector store
        if vector_store_type == 'faiss':
            if index is None or vectors is None:
                logging.error(f"Failed to load/create vector store for '{store_name}'")
            all_chunks.extend(summarized_chunks)
            all_vectors.extend(vectors)
            all_indices.append(index)
    
        elif vector_store_type == 'elasticsearch':
            if all_vectors is None:
                logging.error(f"Failed to load/create vector store for '{store_name}'")        

    
    conn.close()
    return store_name, all_chunks, all_indices, all_vectors
        
def measure_performance(store_name, all_chunks, indices, vectors, query, vector_store_type):
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
    logging.info(f"The vector_store_type: {vector_store_type}")
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
            logging.info(f"I am in ElasticSearch block in measure_performance function")
            body = {
                "query": {
                    "knn": {
                        "field": "vector",
                        "query_vector": query_vector.tolist(),
                        "num_candidates": 3
                    }
                }
            }

            logging.info(f" Starting to Search with index '{store_name}'")
            search_results = es.search(index=store_name, body=body)
            #logging.info(f" Search result '{search_results}' ")
            results.extend(hit['_source']['text'] for hit in search_results['hits']['hits'])

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
    #st.set_page_config(page_title="LLM Chat App", page_icon=":robot_face:", layout="wide")
    
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

    with st.container():
        pdf_files = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)
        if pdf_files:
            with st.spinner('Processing PDFs...'):
                logging.info(f"Calling process_pdfs.")
                store_name, all_chunks, indices, vectors = process_pdfs(pdf_files, language, vector_store_type)

            if np.size(vectors) > 0:  # Correct way to check if vectors array is not empty
                query = st.text_input(f"How can I help you ({language}):")
                logging.info(f"The user Question: {query}")
                if query:
                    results, elapsed_time = measure_performance(store_name, all_chunks, indices, vectors, query, vector_store_type)
                    
                    st.markdown("---")
                    st.subheader("Answer:")
                    st.write(results)
                    st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")
            else:
                logging.info("No vectors found.")
           
def debugfunRemoving():
    srote_name= 'pdftext1'
    pdfname= 'pdftext1'
    document_hash='z8cjvR8OR6OOciQbub1vFg'
    
    removing_vector_elasticsearch(srote_name) 
    #delete_data_from_tables(conn, pdf.name, document_hash) #working
    delete_data_from_tables( pdfname,document_hash)  

if __name__ == '__main__':
    #debugfunRemoving()                                                 
    main()


