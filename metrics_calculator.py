import os
import re
import pandas as pd
import numpy as np 

# --- A. Build TF Matrix ---
def TF_Matrix(df):
    print("\n=======================================================")
    print(" Term Frequency (TF) Calculation")
    print("=======================================================")
    
    # Helper for numerical column sorting (Doc1, Doc2, Doc10...)
    def extract_doc_num(d):
        return int(re.findall(r'\d+', d)[0])

    # 1. Generate TF Matrix (Count frequency of terms per document)
    tf_matrix = pd.crosstab(df['term'], df['doc_id'])
    
    # 2. Sort columns numerically
    sorted_cols = sorted(tf_matrix.columns, key=extract_doc_num)
    tf_matrix = tf_matrix[sorted_cols]
    
    #UPDATE: PRINT FULL TF MATRIX TO TERMINAL ---
    print("\n[Term Frequency (TF) Matrix]:")
    print(tf_matrix.to_string())
    print("-" * 40)
    
    # Save to CSV
    tf_matrix.to_csv('TF_matrix.csv')
    print(">> 'TF_matrix.csv' saved successfully.")
    
    return tf_matrix


# --- B. IDF + TF-IDF Matrix ---
def TF_IDF_Matrix(tf_matrix):
    print("\n=======================================================")
    print(" IDF & TF-IDF Calculation")
    print("=======================================================")
    
    # 1. Calculate N (Total Documents)
    N = len(tf_matrix.columns)
    print(f"Total Documents (N): {N}")
    
    # 2. Calculate DF (Document Frequency)
    df_counts = (tf_matrix > 0).sum(axis=1)
    
    # 3. Calculate IDF
    idf_values = np.log10(N / df_counts)
    
    idf_df = pd.DataFrame(idf_values, columns=['IDF'])
    
    print("\n[Inverse Document Frequency (IDF) Values]:")
    print(idf_df.to_string())
    print("-" * 40)
    
    # Save IDF to CSV
    idf_df.to_csv('IDF_values.csv')
    print(">> 'IDF_values.csv' saved successfully.")
    
    # 4. Create TF-IDF Matrix
    tfidf_matrix = tf_matrix.multiply(idf_values, axis=0)
    
    # --- UPDATE: PRINT FULL TF-IDF MATRIX TO TERMINAL ---
    print("\n[TF-IDF Matrix]:")
    print(tfidf_matrix.to_string())
    print("-" * 40)
    
    # Save TF-IDF to CSV
    tfidf_matrix.to_csv('TFIDF_matrix.csv')
    print(">> 'TFIDF_matrix.csv' saved successfully.")
    
    return tfidf_matrix

if __name__ == "__main__":
    try:
        df_clean = pd.read_csv('cleaned_documents.csv')
        
        # Calculate TF
        tf_matrix = TF_Matrix(df_clean)
        
        # Calculate IDF and TF-IDF
        TF_IDF_Matrix(tf_matrix)
        
        print("\nmetrics_cal.py finished successfully.")
        
    except FileNotFoundError:
        print("ERROR: 'cleaned_documents.csv' not found.")
        print("Please run 'data_index_builder.py' first to generate the input data.")