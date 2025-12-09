import os
import re
from pyspark.sql.functions import (
    input_file_name, regexp_replace, lower, split, 
    posexplode, col, collect_list, concat_ws, sort_array, struct, lit
)
from ir_core_library import get_spark_session

# Preprocessing
def cleanData(spark):
    
    directory = 'project_dataSet-1'
    # Ensure full path or relative path existence
    if not os.path.exists(directory):
        # Fallback to absolute if simple relative fails depending on execution context
        # But 'spark.read' with local[*] handles local files well if path is correct relative to cwd
        print(f"Error: Folder '{directory}' not found.")
        return None

    print(f"Reading files from {directory}...")
    
    # Read all text files from the directory
    # "recursiveFileLookup" is true by default for simple dirs, but explicit glob helps
    df_raw = spark.read.text(os.path.join(directory, "*.txt"))
    
    # Extract filename to get Doc ID
    # input_file_name() returns full path, we need to extract just the name
    df_with_name = df_raw.withColumn("filepath", input_file_name())
    
    # Extract Filename first
    df_with_id = df_with_name.withColumn(
        "filename", 
        regexp_replace(col("filepath"), r"^.*[\\/]", "")
    )
    
    # Extract ID: remove .txt and Prepend "Doc"
    df_with_id = df_with_id.withColumn(
        "doc_id",
        regexp_replace(col("filename"), r"\.txt$", "") 
    ).withColumn(
        "doc_id",
        concat_ws("", lit("Doc"), col("doc_id"))
    )

    # Clean Content
    # 1. Lowercase
    # 2. Remove non-alpha (keep spaces)
    df_cleaned = df_with_id.withColumn("value", lower(col("value"))) \
        .withColumn("value", regexp_replace(col("value"), r'[^a-z\s]', ''))

    # Tokenize and Positional Explode
    # split by space
    df_tokens = df_cleaned.select(
        col("doc_id"), 
        posexplode(split(col("value"), r"\s+")).alias("pos", "term")
    )

    # Filter empty terms
    df_tokens = df_tokens.filter(col("term") != "")
    
    # Adjust position to be 1-indexed
    df_final = df_tokens.withColumn("position", col("pos") + 1).drop("pos")

    # Sort: Term, Doc, Position (For view/csv consistency)
    # Using a udf or just simple sort. Doc ID sort might be string-wise (Doc1, Doc10, Doc2) unless we fix it.
    # We will leave as string sort for now or fix with length.
    df_final = df_final.orderBy("term", "doc_id", "position")

    # Save/Cache
    # We can write to a single CSV for compatibility with other tools if they expected it,
    # or just return the DF. The prompt implies using Spark environment.
    # To keep "cleaned_documents.csv" available for other legacy scripts if any:
    # df_final.toPandas().to_csv('cleaned_documents.csv', index=False)
    
    print("Data cleaned (Spark DataFrame created).")
    return df_final


#  Positional Index Builder
def positionalIndex(df):
    print("\n     Positional Index (Spark)     \n")
    
    # Requirement: < term doc1: pos1, pos2; doc2: pos1... >
    
    # 1. Group by Term, DocID -> Collect positions
    # positions_list should be sorted
    df_doc_positions = df.groupBy("term", "doc_id") \
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
    # We want to sort the docs too. 
    # struct(doc_id, doc_entry) -> sort -> select entry
    df_term_index = df_doc_entry.groupBy("term") \
        .agg(collect_list(struct("doc_id", "doc_entry")).alias("entries_struct"))
        
    # Python-side formatting might be easier for the final string or UDF
    # But let's try native functions. 
    # Sorting array of structs sorts by first field (doc_id). 
    # Note: "Doc10" comes before "Doc2" in string sort. 
    
    # Convert to standard python list to write to file exactly as before (line by line)
    # Collect to driver
    index_rows = df_term_index.collect()
    
    # Sort terms
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
    for line in output_lines[:20]: # Show first 20
        print(line)
    if len(output_lines) > 20: print("...")
    print("[END OF INDEX]\n")

    with open('positional_index.txt', 'w') as f:
        for line in output_lines:
            f.write(line + "\n")
            
    print("positional_index.txt created.")
