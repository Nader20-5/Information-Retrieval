# similarity_ranking.py

import pandas as pd
import matplotlib.pyplot as plt
from ir_core_library import calculate_cosine_similarity, query_to_tfidf_vector, load_tfidf_matrix
import numpy as np

def rank_documents(matched_docs, query_terms, tfidf_matrix):
    """
    Ranks the final set of matched documents based on Cosine Similarity.
    """
    print("\n--- Ranking Documents ---")
    
    # 1. Convert Query to TF-IDF Vector
    # The query vector will be aligned with the rows (terms) of the TFIDF matrix.
    query_vector = query_to_tfidf_vector(query_terms, tfidf_matrix)

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
    for doc_id in matched_docs:
        # Get the document vector (column) from the TFIDF matrix
        # Use .values to get the numpy array representation
        doc_vector = tfidf_matrix[doc_id].values
        
        # Calculate Magnitude (Length) for display
        doc_magnitude = np.linalg.norm(doc_vector)
        
        # Calculate score
        score = calculate_cosine_similarity(query_vector, doc_vector)
        similarity_scores[doc_id] = score
        
        # Optimized Table Row
        print(f"{doc_id:<15} | {doc_magnitude:<25.4f} | {score:.4f}")
        
        # Normalized Vector Output - Formatted nicely
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
    
    # 1. Load dependencies
    tfidf_matrix = load_tfidf_matrix()
    
    if tfidf_matrix is not None:
        # 2. Define a test case (must use terms found in the corpus)
        TEST_MATCHED_DOCS = {'Doc1', 'Doc2', 'Doc4', 'Doc5'} # Documents matching "brutus" AND "mercy"
        TEST_QUERY_TERMS = ['brutus', 'mercy', 'worser'] # Query terms for vector
        
        # 3. Rank
        final_ranking = rank_documents(TEST_MATCHED_DOCS, TEST_QUERY_TERMS, tfidf_matrix)
        
        # 4. Display
        if not final_ranking.empty:
            print("\n--- FINAL RANKED OUTPUT TABLE ---")
            print(final_ranking)
            create_visualizations(final_ranking, tfidf_matrix, TEST_QUERY_TERMS)