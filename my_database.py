import os
import streamlit as st
import sqlite3
import numpy as np


import logging
import time


# Connect to the SQLite database
conn = sqlite3.connect('my_database.db')
c = conn.cursor()

# Function to fetch all data from a table
# Function to fetch all data from a table
def fetch_all_data(table_name):
    c.execute(f'SELECT * FROM {table_name}')
    rows = c.fetchall()
    
    # Get column names
    column_names = [description[0] for description in c.description]
    
    # Print column names
    print(f"Column Names: {', '.join(column_names)}")
    
    # Print rows
    for row in rows:
        print(row)

# Set up logging
log_filename = 'appdb.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),  # Save logs to file
        logging.StreamHandler()  # Also output logs to console
    ]
)

def delete_data_from_tables(conn, pdfname, document_hash):
    try:
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


# Main function
def main():
    # Fetch and print all data from the vector_store table
    print("Vector Store Data:")
    #fetch_all_data('vector_store')
    #delete_data_from_tables(conn, 'pdftext.pdf', '65873b0bf08e035ff2e2c3892aefae8b')
    
    # Fetch and print all data from the document_metadata table
    print("\nDocument Metadata:")
    fetch_all_data('document_metadata')

    # Close the connection
    conn.close()

if __name__ == '__main__':
    main()
