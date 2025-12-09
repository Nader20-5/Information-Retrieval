import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import ALL components
from data_and_index_builder import cleanData, positionalIndex
from metrics_calculator import TF_Matrix, TF_IDF_Matrix
from similarity_ranking import rank_documents, load_tfidf_matrix_spark, create_visualizations
from ir_core_library import load_positional_index_spark, get_spark_session

def run_full_pipeline():

    print("\n\t\tSTARTING FULL INFORMATION RETRIEVAL PIPELINE (SPARK)\t\t\n")
    
    spark = get_spark_session("Full_Pipeline")
    spark.sparkContext.setLogLevel("ERROR")

    # --- Step 1: Data Cleaning & Preprocessing ---
    print("\n>> STEP 1: CLEANING DATA (Spark)...")
    df_clean = cleanData(spark)
    if df_clean is None:
        print("Pipeline aborted at Step 1.")
        return

    # --- Step 2: Positional Index Construction ---
    print("\n>> STEP 2: BUILDING POSITIONAL INDEX (Spark)...")
    positionalIndex(df_clean)

    # --- Step 3: Metrics Calculation (TF, IDF, TF-IDF) ---
    print("\n>> STEP 3: CALCULATING METRICS (TF, IDF, TF-IDF) (Spark)...")
    tf_df = TF_Matrix(df_clean)
    tfidf_df = TF_IDF_Matrix(tf_df)

    # --- Step 4: Search & Ranking Demonstration ---
    print("\n>> STEP 4: DEMONSTRATING RANKING (Spark)...")
    
    # Reload matrices (optional, but validates save/load)
    # tfidf_df = load_tfidf_matrix_spark(spark, "TFIDF_matrix_spark")
    
    if tfidf_df is not None:
        # Define a test query same as the one in similarity_ranking.py
        # Query: "brutus" AND "mercy" -> Matches Doc1, Doc2, Doc4, Doc5 (example from Shakespeare)
        TEST_MATCHED_DOCS = {'Doc1', 'Doc2', 'Doc4', 'Doc5'} 
        TEST_QUERY_TERMS = ['brutus', 'mercy', 'worser'] 
        
        print(f"Test Query Terms: {TEST_QUERY_TERMS}")
        print(f"Matched Documents (Simulated): {TEST_MATCHED_DOCS}")
        
        final_ranking = rank_documents(TEST_MATCHED_DOCS, TEST_QUERY_TERMS, tfidf_df)
        
        if not final_ranking.empty:
            print("\n--- FINAL RANKED OUTPUT TABLE ---")
            print(final_ranking)
            # Visualization is optional
            # create_visualizations(final_ranking, tfidf_df, TEST_QUERY_TERMS)
            
    # Stop Spark? In a real script usually yes.
    # spark.stop()

    print("\n\t\tPIPELINE EXECUTION COMPLETE\t\t\n")

if __name__ == "__main__":
    run_full_pipeline()
