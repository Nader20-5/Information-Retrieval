import os
import re
import pandas as pd
import numpy as np 

# --- A. Build TF Matrix ---
def TF_Matrix(df):
    print("\n\n"+"-"*40)
    print("\n\t\tTerm Frequency (TF) Calculation\t\t\n")
    
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
    print("\n IDF & TF-IDF Calculation\n")
    
    # 1. Calculate N (Total Documents)
    N = len(tf_matrix.columns)
    print(f"\nTotal Documents (N): {N}")
    
    # 2. Calculate DF (Document Frequency)
    df_counts = (tf_matrix > 0).sum(axis=1)
    
    # 3. Calculate IDF
    idf_values = np.log10(N / df_counts)
    
    # Create DF & IDF DataFrame
    idf_df = pd.DataFrame({'DF': df_counts, 'IDF': idf_values})
    print("\n[Document Frequency (DF) & Inverse Document Frequency (IDF)]:\n")
    print(idf_df.to_string())
    print("\n\n"+"-"*40)
    
    # Save IDF to CSV
    idf_df.to_csv('IDF_values.csv')
    print(">> 'IDF_values.csv' saved successfully.")
    
    # 4. Calculate Weighted TF (W-TF) -> 1 + log(TF)
    # Apply 1 + log10(tf) where tf > 0
    w_tf_matrix = tf_matrix.applymap(lambda x: 1 + np.log10(x) if x > 0 else 0)
    print("\n\n"+"-"*40)
    print("\n[Weighted Term Frequency (W-TF) Matrix (1 + log10(tf))]:\n")
    print(w_tf_matrix.to_string())

    # 5. Create TF-IDF Matrix (Using W-TF * IDF is standard, or TF * IDF?)
    tfidf_matrix = tf_matrix.multiply(idf_values, axis=0)
    
    # --- UPDATE: PRINT FULL TF-IDF MATRIX TO TERMINAL ---
    print("\n\n"+"-"*40)
    print("\n[TF-IDF Matrix]:\n")
    print(tfidf_matrix.to_string())

    # 6. Calculate Document Vector Lengths
    # Length = Sqrt(Sum(weight^2)) for each document column
    doc_lengths = np.linalg.norm(tfidf_matrix, axis=0)
    doc_lengths_df = pd.DataFrame(doc_lengths, index=tfidf_matrix.columns, columns=['Vector Length'])
    
    print("\n[Document Vector Lengths]:")
    print(doc_lengths_df.to_string())
    
    # 7. Normalized TF-IDF Matrix
    # Divide each column (document vector) by its length
    # Using broadcasting: divide each column by the corresponding length
    normalized_tfidf_matrix = tfidf_matrix.div(doc_lengths)
    print("\n"+"-"*40)
    print("\n[Normalized TF-IDF Matrix]:\n")
    print(normalized_tfidf_matrix.to_string())
    
    
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