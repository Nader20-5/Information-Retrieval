import os
import re
import math
from pyspark.sql.functions import col, count, lit, log10, sum as _sum, pow, sqrt, round, when
from ir_core_library import get_spark_session

# --- A. Build TF Matrix ---
def TF_Matrix(df_clean):
    print("\n\n"+"-"*40)
    print("\n\t\tTerm Frequency (TF) Calculation (Spark)\t\t\n")
    
    # 1. Generate TF Matrix (Count frequency of terms per document)
    # Pivot doc_id to columns
    # Group by term
    
    # Get all doc_ids to ensure pivot covers them all (even if missing in some)
    # Though pivot does this dynamically.
    
    tf_df = df_clean.groupBy("term") \
        .pivot("doc_id") \
        .count() \
        .na.fill(0)
        
    # Sort columns numerically for display consistency (Doc1, Doc2, Doc10)
    # Spark columns are not ordered by default like Pandas, so we select.
    doc_cols = [c for c in tf_df.columns if c != "term"]
    
    def extract_doc_num(d):
        try:
            return int(re.findall(r'\d+', d)[0])
        except:
            return 0
            
    sorted_doc_cols = sorted(doc_cols, key=extract_doc_num)
    tf_df = tf_df.select("term", *sorted_doc_cols).orderBy("term")
    
    #UPDATE: PRINT EXCERPT OF TF MATRIX TO TERMINAL ---
    print("\n[Term Frequency (TF) Matrix - Top 20 rows]:")
    tf_df.show(20, truncate=False)
    print("-" * 40)
    
    # Save to CSV (Coalesce to 1 to produce single file for easy inspection)
    # output path directory
    output_path = "TF_matrix_spark"
    tf_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
    print(f">> '{output_path}' saved successfully.")
    
    return tf_df

# --- B. IDF + TF-IDF Matrix ---
def TF_IDF_Matrix(tf_df):
    print("\n IDF & TF-IDF Calculation (Spark)\n")
    
    doc_cols = [c for c in tf_df.columns if c != "term"]
    N = len(doc_cols)
    print(f"\nTotal Documents (N): {N}")
    
    # 2. Calculate DF (Document Frequency)
    df_expr = sum([when(col(c) > 0, 1).otherwise(0) for c in doc_cols])
    
    df_idf_calc = tf_df.withColumn("DF", df_expr)
    
    # 3. Calculate IDF = log10(N / DF)
    df_idf_calc = df_idf_calc.withColumn("IDF", log10(lit(N) / col("DF")))
    
    # Select only DF and IDF for display/saving
    idf_display = df_idf_calc.select("term", "DF", "IDF").orderBy("term")
    
    print("\n[Document Frequency (DF) & Inverse Document Frequency (IDF)]:\n")
    idf_display.show(20, truncate=False)
    print("\n\n"+"-"*40)
    
    # Save IDF
    idf_display.coalesce(1).write.mode("overwrite").option("header", "true").csv("IDF_values_spark")
    print(">> 'IDF_values_spark' saved successfully.")
    
    # 4. Calculate Weighted TF (W-TF) -> 1 + log10(TF)
    print("\n[Weighted Term Frequency (W-TF) Matrix (1 + log10(tf))]:\n")
    
    w_tf_cols = [col("term")]
    for doc in doc_cols:
        # 1 + log10(tf) if tf > 0 else 0
        w_tf_cols.append(
            when(col(doc) > 0, 1 + log10(col(doc))).otherwise(0).alias(doc)
        )
        
    w_tf_df = tf_df.select(*w_tf_cols).orderBy("term")
    w_tf_df.show(20, truncate=False)
    
    # Save W-TF Matrix
    w_tf_df.coalesce(1).write.mode("overwrite").option("header", "true").csv("WTF_matrix_spark")
    print(">> 'WTF_matrix_spark' saved successfully.")

    # 5. Create TF-IDF Matrix (W-TF * IDF)
    # Join w_tf_df with idf_display to get IDF column
    
    w_tf_with_idf = w_tf_df.join(idf_display.select("term", "IDF"), on="term", how="inner")
    
    tfidf_cols = [col("term")]
    for doc in doc_cols:
        tfidf_cols.append((col(doc) * col("IDF")).alias(doc))
        
    tfidf_matrix = w_tf_with_idf.select(*tfidf_cols).orderBy("term")
    
    print("\n\n"+"-"*40)
    print("\n[TF-IDF Matrix]:\n")
    tfidf_matrix.show(20, truncate=False)

    # 6. Calculate Document Vector Lengths
    doc_lengths = {}
    for doc in doc_cols:
        length = tfidf_matrix.select(sqrt(_sum(pow(col(doc), 2)))).collect()[0][0]
        doc_lengths[doc] = length
        
    # Display Lengths
    print("\n[Document Vector Lengths]:")
    for doc, length in doc_lengths.items():
        print(f"{doc}: {length}")
        
    # 7. Normalized TF-IDF Matrix
    norm_cols = [col("term")]
    for doc in doc_cols:
        length = doc_lengths.get(doc, 1.0)
        length = length if length > 0 else 1.0
        norm_cols.append((col(doc) / length).alias(doc))
        
    normalized_tfidf_matrix = tfidf_matrix.select(*norm_cols)
    
    print("\n"+"-"*40)
    print("\n[Normalized TF-IDF Matrix]:\n")
    normalized_tfidf_matrix.show(20, truncate=False)
    
    # Save to CSV
    tfidf_matrix.coalesce(1).write.mode("overwrite").option("header", "true").csv("TFIDF_matrix_spark")
    print(">> 'TFIDF_matrix_spark' saved successfully.")
    
    return tfidf_matrix

if __name__ == "__main__":
    # Test execution
    try:
        spark = get_spark_session("Metrics_Test")
        # Load cleaned data ? Or just rely on existing csv? 
        # Ideally we run from 'run_all.py'
        print("Run 'run_all.py' to execute the full pipeline.")
        
    except FileNotFoundError:
        print("ERROR")
