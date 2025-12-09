import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import ALL components
from data_and_index_builder import cleanData, positionalIndex
from metrics_calculator import TF_Matrix, TF_IDF_Matrix
from similarity_ranking import rank_documents, load_tfidf_matrix, create_visualizations
from ir_core_library import load_positional_index

def run_full_pipeline():

    print("\n\t\tSTARTING FULL INFORMATION RETRIEVAL PIPELINE\t\t\n")
    

    # --- Step 1: Data Cleaning & Preprocessing ---
    print("\n>> STEP 1: CLEANING DATA...")
    df_clean = cleanData()
    if df_clean is None:
        print("Pipeline aborted at Step 1.")
        return

    # --- Step 2: Positional Index Construction ---
    print("\n>> STEP 2: BUILDING POSITIONAL INDEX...")
    positionalIndex(df_clean)

    # --- Step 3: Metrics Calculation (TF, IDF, TF-IDF) ---
    print("\n>> STEP 3: CALCULATING METRICS (TF, IDF, TF-IDF)...")
    tf_matrix = TF_Matrix(df_clean)
    tfidf_matrix = TF_IDF_Matrix(tf_matrix)

    # --- Step 4: Search & Ranking Demonstration ---
    print("\n>> STEP 4: DEMONSTRATING RANKING...")
    
    # Reload matrices to ensure consistency (optional, but good practice)
    tfidf_matrix = load_tfidf_matrix()
    
    if tfidf_matrix is not None:
        # Define a test query same as the one in similarity_ranking.py
        # Query: "brutus" AND "mercy" -> Matches Doc1, Doc2, Doc4, Doc5 (example)
        TEST_MATCHED_DOCS = {'Doc1', 'Doc2', 'Doc4', 'Doc5'} 
        TEST_QUERY_TERMS = ['brutus', 'mercy', 'worser'] 
        
        print(f"Test Query Terms: {TEST_QUERY_TERMS}")
        print(f"Matched Documents ( Simulated): {TEST_MATCHED_DOCS}")
        
        final_ranking = rank_documents(TEST_MATCHED_DOCS, TEST_QUERY_TERMS, tfidf_matrix)
        
        if not final_ranking.empty:
            print("\n--- FINAL RANKED OUTPUT TABLE ---")
            print(final_ranking)
            # Visualization is optional in a batch script, uncomment if needed
            # create_visualizations(final_ranking, tfidf_matrix, TEST_QUERY_TERMS)
            

    print("\n\t\tPIPELINE EXECUTION COMPLETE\t\t\n")

if __name__ == "__main__":
    run_full_pipeline()
