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
#  [Spark Implementation for Index Only]
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, sort_array, concat_ws, struct, col

def positionalIndex(df):
    print("\n     Positional Index (Spark)     \n")
    
    # Initialize Spark Session locally just for this function
    spark = SparkSession.builder \
        .appName("PositionalIndexBuilder") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    # Check if df is empty
    if df.empty:
        print("Dataframe is empty, skipping index.")
        return

    # Convert Pandas DataFrame to Spark DataFrame
    # Ensure columns match expectations: term, doc_id, position
    sdf = spark.createDataFrame(df)
    
    # Requirement: < term doc1: pos1, pos2; doc2: pos1... >
    
    # 1. Group by Term, DocID -> Collect positions
    # positions_list should be sorted
    # Note: sort_array guarantees order
    df_doc_positions = sdf.groupBy("term", "doc_id") \
        .agg(sort_array(collect_list("position")).alias("positions"))
    
    # 2. Format positions to string "pos1, pos2"
    df_doc_str = df_doc_positions.withColumn(
        "pos_str", concat_ws(", ", col("positions").cast("array<string>"))
    )
    
    # 3. Format doc entry "Doc1: pos1, pos2"
    df_doc_entry = df_doc_str.withColumn(
        "doc_entry", 
        concat_ws(": ", col("doc_id"), col("pos_str"))
    )
    
    # 4. Group by Term -> Collect doc entries
    # structure the doc_entry with doc_id so we can sort by doc_id later
    df_term_index = df_doc_entry.groupBy("term") \
        .agg(collect_list(struct("doc_id", "doc_entry")).alias("entries_struct"))
    
    # Collect to driver to format final output strings (compatible with legacy text format)
    index_rows = df_term_index.collect()
    
    # Stop Spark as we are done with the "Index uses Spark" part
    spark.stop()
    
    # Sort terms A-Z
    index_rows.sort(key=lambda row: row['term'])
    
    output_lines = []
    
    # Custom sort for Doc IDs (numeric)
    def extract_doc_num(d):
        try:
            return int(re.findall(r'\d+', d)[0])
        except:
            return 0

    for row in index_rows:
        term = row['term']
        entries = row['entries_struct'] # List of Rows(doc_id, doc_entry)
        
        # Sort entries by Doc ID numerically
        entries.sort(key=lambda x: extract_doc_num(x.doc_id))
        
        doc_strings = [x.doc_entry for x in entries]
        full_string = "; ".join(doc_strings)
        output_lines.append(f"< {term} {full_string} >")

    print("\n[DISPLAYING POSITIONAL INDEX]")
    for line in output_lines[:20]:
        print(line)
    if len(output_lines) > 20: print("...")
    print("[END OF INDEX]\n")

    with open('positional_index.txt', 'w') as f:
        for line in output_lines:
            f.write(line + "\n")
            
    print("positional_index.txt created (via Spark).")