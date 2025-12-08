import os
import re
import pandas as pd
import numpy as np 

# Preprocessing
def cleanData():
    
    directory = 'project_dataSet-1'
    data = []

    if not os.path.exists(directory):
        print(f"Error: Folder '{directory}' not found. Please create it and add .txt files.")
        return None

    files = sorted(os.listdir(directory))
    for filename in files:
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            
            # Read content (Handle different encodings)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
            except:
                with open(filepath, 'r', encoding='cp1252') as file:
                    content = file.read()
            
            #Cleaning
            content = content.lower()
            content = re.sub(r'[^a-z\s]', '', content)

            #Tokenization
            tokens = content.split()
            
            # Extract Doc ID safely
            doc_nums = re.findall(r'\d+', filename)
            if doc_nums:
                doc_id = "Doc" + doc_nums[0]
            else:
                doc_id = "Doc" + filename.replace('.txt', '')
            
            for index, word in enumerate(tokens):
                data.append({
                    "doc_id": doc_id,
                    "term": word,
                    "position": index + 1
                })

    df = pd.DataFrame(data)
    #Sort: Term A-Z, then Doc1-10, then Position 1-N
    df = df.sort_values(by=['term', 'doc_id', 'position'])
    
    df.to_csv('cleaned_documents.csv', index=False)
    print("Data cleaned")
    return df


#  Positional Index Builder
def positionalIndex(df):
    print("\n ------------Positional Index------------ ")
    
    # Requirement: < term doc1: pos1, pos2; doc2: pos1... >
    grouped = df.groupby(['term', 'doc_id'])['position'].apply(list).reset_index()
    output_lines = []
    
    # Helper to sort "Doc1, Doc2, Doc10" correctly
    def extract_doc_num(d):
        return int(re.findall(r'\d+', d)[0])

    for term in sorted(df['term'].unique()):
        term_data = grouped[grouped['term'] == term]
        doc_strings = []
        
        sorted_docs = sorted(term_data['doc_id'].unique(), key=extract_doc_num)
        
        for doc in sorted_docs:
            positions = term_data[term_data['doc_id'] == doc]['position'].values[0]
            pos_str = ", ".join(map(str, positions))
            doc_strings.append(f"{doc}: {pos_str}")
        
        full_string = "; ".join(doc_strings)
        output_lines.append(f"< {term} {full_string} >")

    print("\n[DISPLAYING POSITIONAL INDEX]")
    for line in output_lines:
        print(line)
    print("[END OF INDEX]\n")

    with open('positional_index.txt', 'w') as f:
        for line in output_lines:
            f.write(line + "\n")
            
    print("positional_index.txt created.")