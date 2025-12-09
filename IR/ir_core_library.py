import re
import numpy as np
import pandas as pd

def load_positional_index(file_path='positional_index.txt'):
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
        print(f"Error: Positional index file not found at {file_path}. Run 01_data_and_index_builder.py first.")
        return None

    return positional_index

def match_phrase_terms(terms_list, index):
    num_terms = len(terms_list)
    if num_terms == 0:
        return set()
    
    #Handle Single Term Case (Base Case)
    term0 = terms_list[0]
    if term0 not in index:
        return set()
        
    if num_terms == 1:
        return set(index[term0].keys())

    current_doc_postings = index[term0]
    
    for i in range(1, num_terms):
        term_prev = terms_list[i - 1] #The term whose position we use as a reference
        term_current = terms_list[i]  #The term we are checking the position of
        
        if term_current not in index:
            return set()
            
        postings_current = index[term_current]
        new_doc_matches = {}
        
        # Find the intersection of documents: those from the previous match AND those 
        # that contain the current term.
        common_docs = set(current_doc_postings.keys()) & set(postings_current.keys())

        for doc_id in common_docs:
            p_prev = current_doc_postings[doc_id] # Positions of the matched sequence start (or just term_prev)
            p_current = postings_current[doc_id] # Positions of the current term
            
            matched_positions = []
            
            for start_pos_seq in p_prev: 
                required_pos_current = start_pos_seq + i 
        
                if required_pos_current in p_current:
                    matched_positions.append(start_pos_seq)
            
            if matched_positions:
                new_doc_matches[doc_id] = matched_positions
        
        if not new_doc_matches:
            return set()
            
        # The result of this step becomes the input for the next term's check.
        current_doc_postings = new_doc_matches

    return set(current_doc_postings.keys())

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

# --- F. Ranking and Vectorization ---

def load_tfidf_matrix(file_path='TFIDF_matrix.csv'):
    try:
        # We set 'term' as the index so we can look up term vectors easily
        return pd.read_csv(file_path, index_col='term')
    except FileNotFoundError:
        print(f"Error: TFIDF matrix file not found at {file_path}. Run metrics_cal.py first.")
        return None

def query_to_tfidf_vector(query_terms, tfidf_matrix):
    
    corpus_terms = tfidf_matrix.index
    
    # We create a vector whose length is the size of the vocabulary
    query_vector = np.zeros(len(corpus_terms))
    
    for term in set(query_terms):
        if term in corpus_terms:
            term_index = corpus_terms.get_loc(term)
            query_vector[term_index] = 1 
            
    return query_vector

def calculate_cosine_similarity(query_vector, doc_vector):
    # Ensure vectors are numpy arrays for dot product
    q_vec = np.array(query_vector)
    d_vec = np.array(doc_vector)
    
    # Dot Product (Numerator)
    dot_product = np.dot(q_vec, d_vec)
    
    # Magnitude Calculation
    q_magnitude = np.linalg.norm(q_vec)
    d_magnitude = np.linalg.norm(d_vec)
    
    # Avoid division by zero
    if q_magnitude == 0 or d_magnitude == 0:
        return 0.0
    
    # Cosine Similarity Formula
    return dot_product / (q_magnitude * d_magnitude)