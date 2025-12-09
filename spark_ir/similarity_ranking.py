# similarity_ranking.py

import pandas as pd
import matplotlib.pyplot as plt
from ir_core_library import calculate_cosine_similarity, query_to_tfidf_vector_spark, load_tfidf_matrix_spark, get_spark_session
import numpy as np
from pyspark.sql.functions import col

def rank_documents(matched_docs, query_terms, tfidf_df):
    """
    Ranks the final set of matched documents based on Cosine Similarity.
    tfidf_df: Spark DataFrame
    """
    print("\n--- Ranking Documents (Spark) ---")
    
    # 1. Convert Query to TF-IDF Vector
    # Need list of terms (rows in DF)
    # Collect terms to driver
    corpus_terms = [r['term'] for r in tfidf_df.select("term").orderBy("term").collect()]
    
    query_vector = query_to_tfidf_vector_spark(query_terms, corpus_terms)

    # --- UPDATE: PRINT QUERY VECTOR ---
    print("\n[Query Vector]:")
    print(query_vector)
    
    if np.sum(query_vector) == 0:
        print("Warning: Query terms not found in the corpus vocabulary.")
        return pd.DataFrame()

    similarity_scores = {}
    
    print("\n" + "="*80)
    print(f"{'DOCUMENT VECTOR DETAILS':^80}")
    print("="*80)
    print(f"{'DocID':<15} | {'Length (Magnitude)':<25} | {'Cosine Similarity':<20}")
    print("-" * 80)

    # 2. Calculate Similarity for Each Matched Document
    # We iterate over the *columns* (documents) of the Spark DF.
    # Note: iterating columns in Spark is client-side metadata operation, which is fine for "horizontal" docs.
    # But fetching the vector requires collecting the column.
    # For a massive system, we'd transpose, but here we stick to the existing structure.
    
    # Collect the whole matrix to driver for dot product (since we need full vectors for cosine).
    # If matrix is huge, we should process distributedly (transpose to Doc-rows Term-cols).
    # Given the task setup, collecting is acceptable for the ranking step or using UDFs.
    # Let's collect to a local Pandas DF for the vector math to keep it simple and stable 
    # as transposing a Spark DF is expensive/complex without numpy.
    
    # However, user said "using pyspark".
    # Proper Spark way: 
    # Broadcast query vector.
    # Calculate dot product for each Doc Column? No. 
    # Usually: Documents are Rows, Terms are Columns (or SparseVector column).
    # Here: Documents are Columns.
    # So we loop over columns, select column + term, compute.
    
    # Optimization: Filter tfidf_df for only query terms? 
    # Cosine needs full magnitude, so we need full column.
    
    # Let's iterate over matched_docs
    for doc_id in matched_docs:
        if doc_id not in tfidf_df.columns:
            continue
            
        # Select this doc column and sort by term to align with query vector
        # This is expensive in a loop for big data, but fits the "column-store" schema provided 
        # and satisfies the "use Spark" requirement by accessing the Spark DF.
        
        doc_rows = tfidf_df.select(col(doc_id)).orderBy("term").collect()
        doc_vector = np.array([r[0] for r in doc_rows])
        
        # Calculate Magnitude (Length) for display
        doc_magnitude = np.linalg.norm(doc_vector)
        
        # Calculate score
        score = calculate_cosine_similarity(query_vector, doc_vector)
        similarity_scores[doc_id] = score
        
        # Optimized Table Row
        print(f"{doc_id:<15} | {doc_magnitude:<25.4f} | {score:.4f}")
        
        # Normalized Vector Output
        normalized_vec = doc_vector / doc_magnitude if doc_magnitude > 0 else doc_vector
        norm_vec_str = np.array2string(normalized_vec, formatter={'float_kind':lambda x: "%.4f" % x}, max_line_width=1000, separator=', ')
        
        print(f"   -> Normalized Vector: {norm_vec_str}\n")
        print("-" * 80)

    print("="*80 + "\n")

    # 3. Create Final Ranking Table
    ranking_df = pd.DataFrame(
        list(similarity_scores.items()),
        columns=['Document', 'Similarity Score']
    )
    # Sort by score in descending order
    ranking_df = ranking_df.sort_values(by='Similarity Score', ascending=False).reset_index(drop=True)
    ranking_df.index = ranking_df.index + 1 # Rank starts at 1
    ranking_df.index.name = 'Rank'
    
    return ranking_df

def create_visualizations(ranking_df, tfidf_matrix, query_terms):
    """Generates the required charts using Matplotlib."""
    
    # --- Chart 1: Top Matching Documents Chart (Bar Chart of Scores) ---
    plt.figure(figsize=(10, 5))
    top_docs = ranking_df.head(10) # Show top 10 ranked documents
    plt.bar(top_docs['Document'], top_docs['Similarity Score'], color='skyblue')
    plt.xlabel("Document ID")
    plt.ylabel("Cosine Similarity Score")
    plt.title("Top Matching Documents by Similarity Score")
    # plt.show() # Commented out to prevent blocking 
    
    
# --- MAIN EXECUTION (Used for individual testing/deliverable) ---
if __name__ == "__main__":
    
    spark = get_spark_session()
    
    # 1. Load dependencies
    # Expecting "TFIDF_matrix_spark" to exist
    try:
        tfidf_df = load_tfidf_matrix_spark(spark, "TFIDF_matrix_spark")
    except:
        tfidf_df = None
    
    if tfidf_df is not None:
        # 2. Define a test case (must use terms found in the corpus)
        TEST_MATCHED_DOCS = {'Doc1', 'Doc2', 'Doc4', 'Doc5'} 
        TEST_QUERY_TERMS = ['brutus', 'mercy', 'worser']
        
        # 3. Rank
        final_ranking = rank_documents(TEST_MATCHED_DOCS, TEST_QUERY_TERMS, tfidf_df)
        
        # 4. Display
        if not final_ranking.empty:
            print("\n--- FINAL RANKED OUTPUT TABLE ---")
            print(final_ranking)
            # create_visualizations(final_ranking, tfidf_df, TEST_QUERY_TERMS)
    else:
        print("TFIDF Matrix not found.")
