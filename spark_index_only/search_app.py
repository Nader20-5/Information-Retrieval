import re
import pandas as pd 
from similarity_ranking import rank_documents, create_visualizations
from ir_core_library import (
    load_positional_index, 
    match_phrase_terms,
    boolean_and, 
    boolean_or,   
    boolean_and_not, 
    normalize_phrase
)
from ir_core_library import (
    load_tfidf_matrix 
)


def get_all_doc_ids(index):
    all_docs = set()
    for term in index:
        all_docs.update(index[term].keys())
    return all_docs


def main_search_loop(index ,tfidf_matrix):
    
    print("\n(Phrase Query Processor & Boolean Logic)\n")
    
    all_docs = get_all_doc_ids(index)

    while True:
        user_query = input("\nEnter query (use OR, AND, AND NOT, or NOT [phrase]. Type 'exit' to quit): ")
        
        if user_query.strip().lower() == 'exit':
            print("\nExiting search application. Goodbye!")
            break
        
        
        phrases = re.findall(r'"([^"]*)"', user_query)
        operators = re.findall(r'AND NOT|AND|OR|NOT', user_query) 
        
        if not phrases:
            print("Error: No quoted phrases found in query.")
            continue
        
        
        starts_with_not = user_query.strip().upper().startswith('NOT "')
        
        current_doc_set = set()
        
        first_phrase_terms = normalize_phrase(phrases[0]).split()
        
        initial_match_set = match_phrase_terms(first_phrase_terms, index)
        
        if starts_with_not:
            current_doc_set = all_docs.difference(initial_match_set)
            print(f"\n[Operand 1: NOT '{phrases[0]}'] Matched documents: {current_doc_set}")
            operators = operators[1:] 
        else:
            current_doc_set = initial_match_set
            print(f"\n[Operand 1: '{phrases[0]}'] Matched documents: {current_doc_set}")

        
        for i, operator in enumerate(operators):
            phrase_index = 1 if starts_with_not else i + 1
            if phrase_index >= len(phrases): break
                
            next_phrase_terms = normalize_phrase(phrases[phrase_index]).split()
            
            next_doc_set = match_phrase_terms(next_phrase_terms, index)

            if operator == 'AND':
                current_doc_set = boolean_and(current_doc_set, next_doc_set)
            elif operator == 'OR':
                current_doc_set = boolean_or(current_doc_set, next_doc_set)
            elif operator == 'AND NOT':
                current_doc_set = boolean_and_not(current_doc_set, next_doc_set)
            
            print(f"[{operator} with Operand {phrase_index + 1}] New Matched Documents: {current_doc_set}")
            
        matched_docs = current_doc_set
        
        if matched_docs:
            print("\n--- RESULTS FOR RANKING ---")
            print(f"Matched Documents: {matched_docs}")

            all_query_phrases = [normalize_phrase(p).split() for p in phrases]
            all_query_terms = [term for phrase in all_query_phrases for term in phrase]
            final_ranking_df = rank_documents(set(matched_docs), all_query_terms, tfidf_matrix)

            if not final_ranking_df.empty:
                # Display final table
                print("\n--- FINAL RANKED OUTPUT TABLE ---")
                print(final_ranking_df)

                create_visualizations(final_ranking_df, tfidf_matrix, all_query_terms)
            else:
                print("Matching was successful, but ranking returned empty (possibly due to missing terms in TF-IDF matrix).")
        else:
            print("\nQuery executed successfully, but no documents matched the criteria.")


if __name__ == "__main__":
    # 1. Load data
    positional_index = load_positional_index()
    tfidf_matrix = load_tfidf_matrix()
    
    if positional_index is not None:
        # 2. Start the search loop
        main_search_loop(positional_index , tfidf_matrix)