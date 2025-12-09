import re
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

def get_spark_session(app_name="IR_System"):
    """
    Creates or retrieves a SparkSession.
    """
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()

# --- Shared Logic ---

def normalize_phrase(phrase):
    phrase = phrase.lower()
    phrase = re.sub(r'[^a-z\s]', '', phrase)
    return phrase

def boolean_and(set_a, set_b):
    return set_a.intersection(set_b)

def boolean_or(set_a, set_b):
    return set_a.union(set_b)

def boolean_and_not(set_a, set_b):
    return set_a.difference(set_b)

# --- Spark-based Loaders (Placeholders/Helpers) ---

def load_positional_index_spark(spark, file_path='positional_index.txt'):
    # In a real Spark app, we might just re-compute or load from parquet.
    # For now, we keep the dictionary logic for the search app if we want to keep it simple,
    # OR we can load the index into a DataFrame if we want to be fully "Spark".
    # Given the previous implementation used a dict for 'match_phrase_terms',
    # we can stick to reading the text file into a dict for the *Search App* 
    # as latency for Spark query on small data might be high.
    # However, let's try to be consistent with the user verification.
    
    # Re-using the existing loader for the Search App's in-memory index 
    # (since search is often real-time and Spark might be overkill for a simple interactive loop on local).
    # BUT, the user asked to "do it using pyspark".
    # So we will implement a reader that reads the cleaning output or index output using Spark if needed.
    
    # For the search loop specifically `match_phrase_terms`, it relies on a Dictionary structure.
    # We will keep the `load_positional_index` text-reader for compatibility with the search logic
    # unless we rewrite the entire search logic to be DataFrame-based (which is slower for interactive CLI).
    # Let's keep the file-reader for the dictionary to support the existing search algorithm,
    # but the creation of that file will be done by Spark.
    
    positional_index = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip().strip('<>').strip()
                parts = line.split(' ', 1)
                if len(parts) < 2: continue

                term = parts[0].strip()
                postings_string = parts[1].strip()
                postings_dict = {}
                
                doc_entries = postings_string.split(';')
                
                for entry in doc_entries:
                    if ':' in entry:
                        doc_id, pos_str = entry.split(':', 1)
                        doc_id = doc_id.strip()
                        positions = [int(p.strip()) for p in pos_str.split(',')]
                        postings_dict[doc_id] = positions
                
                positional_index[term] = postings_dict
    except FileNotFoundError:
        print(f"Error: Positional index file not found at {file_path}.")
        return None
    return positional_index

def match_phrase_terms(terms_list, index):
    # This logic remains the same as it operates on the standard Python Dictionary 
    # constructed from the index. Data *generation* is Spark, but *retrieval* here is CPU-bound on the index.
    num_terms = len(terms_list)
    if num_terms == 0:
        return set()
    
    term0 = terms_list[0]
    if term0 not in index:
        return set()
        
    if num_terms == 1:
        return set(index[term0].keys())

    current_doc_postings = index[term0]
    
    for i in range(1, num_terms):
        term_prev = terms_list[i - 1]
        term_current = terms_list[i]
        
        if term_current not in index:
            return set()
            
        postings_current = index[term_current]
        new_doc_matches = {}
        
        common_docs = set(current_doc_postings.keys()) & set(postings_current.keys())

        for doc_id in common_docs:
            p_prev = current_doc_postings[doc_id]
            p_current = postings_current[doc_id]
            
            matched_positions = []
            
            for start_pos_seq in p_prev: 
                required_pos_current = start_pos_seq + i 
        
                if required_pos_current in p_current:
                    matched_positions.append(start_pos_seq)
            
            if matched_positions:
                new_doc_matches[doc_id] = matched_positions
        
        if not new_doc_matches:
            return set()
            
        current_doc_postings = new_doc_matches

    return set(current_doc_postings.keys())

# --- Vector / Ranking Helpers handled in Spark usually, but for Search Loop: ---

def load_tfidf_matrix_spark(spark, file_path='TFIDF_matrix.csv'):
    # Load as Spark DataFrame
    try:
        return spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
    except Exception as e:
        print(f"Error loading TFIDF matrix: {e}")
        return None

def calculate_cosine_similarity(query_vector, doc_vector):
    # Generic math function, compatible with standard lists/arrays
    q_vec = np.array(query_vector)
    d_vec = np.array(doc_vector)
    dot_product = np.dot(q_vec, d_vec)
    q_magnitude = np.linalg.norm(q_vec)
    d_magnitude = np.linalg.norm(d_vec)
    if q_magnitude == 0 or d_magnitude == 0:
        return 0.0
    return dot_product / (q_magnitude * d_magnitude)

def query_to_tfidf_vector_spark(query_terms, corpus_terms):
    # corpus_terms: list of terms (columns from the DF)
    # Returns a list/array representing the query vector
    query_vector = np.zeros(len(corpus_terms))
    for term in set(query_terms):
        if term in corpus_terms:
            term_index = corpus_terms.index(term)
            query_vector[term_index] = 1 
    return query_vector
