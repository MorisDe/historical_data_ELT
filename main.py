import numpy as np
import pandas as pd
from datetime import datetime
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from rapidfuzz import process, fuzz
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import time
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("groq_api_key")

def fix_strings(x):
    if pd.isna(x):
        return None
    
    x=str(x).strip().lower()
    x = re.sub(r'[^\w\s]', '', x) 
    x = re.sub(r'\s+', ' ', x)
    return x

def distance_matrix(full_name):
    n = len(full_name)
    distances = []
    
    for i in range(n):
        if i % 100 == 0:
            print(f"Progress: {i}/{n}")
        for j in range(i+1, n):
            similarity = fuzz.token_sort_ratio(full_name[i], full_name[j])
            distance = 100 - similarity
            distances.append(distance)
        
    distance_matrix = squareform(distances)
    distance_matrix = distance_matrix.astype(np.float32, copy=False)
    
    return distance_matrix

def hierarchical_cluster(distance_matrix,full_name):
    
    clustering = AgglomerativeClustering(
    n_clusters=None, 
    distance_threshold=20, 
    metric='precomputed',
    linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)
    
    from collections import defaultdict
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(full_name[idx])
        
    return clusters
        
def resolve_with_transformer(ner_pipeline,name_list):
    """Use transformer NER model"""
    scores = {}
    
    for name in name_list:
        score = 0
        
        # Get NER predictions
        try:
            entities = ner_pipeline(name)
            for ent in entities:
                if ent['entity_group'] == 'PER':  
                    score += ent['score'] * 20 
        except:
            pass
        
        # Additional heuristics
        score += len(name)
        score += name.count(' ') * 5
        
        # Penalize typos
        typo_chars = ['z', 'x', 'q']
        for char in typo_chars:
            if name.lower().count(char) > 1:
                score -= 5
        
        # Check for complete first and last name
        parts = name.split()
        if len(parts) >= 2 and all(len(p) > 2 for p in parts):
            score += 10
            
        scores[name] = score
    
    best_name = max(scores.items(), key=lambda x: x[1])
    confidence = best_name[1]
    
    return best_name[0], confidence
       
def name_recognized(ner_pipeline,clusters):
    auto_resolved = {}
    ambiguous = {}
    threshold = 25
    
    for cluster_id, name_list in tqdm(clusters.items(), desc="Resolving with Transformer"):
        if len(name_list) == 1:
            continue
        
        resolved_name, confidence = resolve_with_transformer(ner_pipeline,name_list)
        
        if confidence > threshold:
            auto_resolved[cluster_id] = {
                'variations': name_list,
                'resolved': resolved_name,
                'confidence': confidence
            }
        else:
            ambiguous[cluster_id] = name_list
    
    return auto_resolved,ambiguous
    
def low_confidence_name(auto_resolved):
    filtered_auto_resolved = {}
    move_to_llm = {}
    
    for cluster_id, data in auto_resolved.items():
        confidence = data['confidence']
        
        # Check if confidence is np.float32/float64 type (these seem problematic)
        if isinstance(confidence, (np.float32, np.float64)):
            move_to_llm[cluster_id] = data['variations']
        else:
            filtered_auto_resolved[cluster_id] = data
    
    return  filtered_auto_resolved,move_to_llm
           
def llm_component(ambiguous):
    
    groq_api_key=api_key
    chat_model = ChatGroq(
    groq_api_key=groq_api_key,
    # model_name="openai/gpt-oss-120b",#Incase of limit hit use this
    model_name="llama-3.3-70b-versatile", 
    temperature=0.2,
    max_retries=3
    )
    
    batch_name_identification = PromptTemplate.from_template(
    """
    You are given multiple groups of name variations. Each group represents different spellings of the same person's name.
    For each group, determine the most likely correct, fully spelled, standard version.

    Rules:
    - Output the resolved names in this exact format: CLUSTER_ID: resolved_name
    - One line per cluster
    - Output ONLY the cluster ID and resolved name, nothing else
    - No explanations, no extra text, no markdown formatting

    Name groups:
    {name_groups}
    """
    )

    chain = batch_name_identification | chat_model
    
    llm_resolved = {}
    failed_batches = []
    BATCH_SIZE = 70 
    
    ambiguous_items = list(ambiguous.items())
    total_batches = (len(ambiguous_items) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in tqdm(range(0, len(ambiguous_items), BATCH_SIZE), desc="Processing batches"):
        batch = ambiguous_items[batch_idx:batch_idx + BATCH_SIZE]
        
        # Format batch
        name_groups = []
        for cluster_id, name_list in batch:
            variations = ", ".join(name_list)
            name_groups.append(f"CLUSTER_{cluster_id}: {variations}")
        
        batch_input = "\n\n".join(name_groups)
        
        try:
            # Call LLM
            result = chain.invoke({"name_groups": batch_input})
            response = result.content if hasattr(result, 'content') else str(result)
            
            # Parse response
            for line in response.strip().split('\n'):
                line = line.strip()
                if not line or ':' not in line:
                    continue
                    
                try:
                    # Extract cluster ID and resolved name
                    parts = line.split(':', 1)
                    cluster_part = parts[0].strip().replace('CLUSTER_', '').replace('np.int64(', '').replace(')', '')
                    resolved_name = parts[1].strip()
                    
                    # Convert to int
                    cluster_id = int(cluster_part)
                    
                    # Find original name list
                    original_names = None
                    for cid, names in batch:
                        if cid == cluster_id:
                            original_names = names
                            break
                    
                    if original_names:
                        llm_resolved[cluster_id] = {
                            'variations': original_names,
                            'resolved': resolved_name
                        }
                except Exception as e:
                    print(f"\nParsing error for line: {line[:50]}... - {e}")
                    continue
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\nError in batch {batch_idx//BATCH_SIZE}: {e}")
            failed_batches.append(batch)
            continue


    return  llm_resolved      

def fill_missing_by_group(df, group_col, target_col):

    df[target_col] = df[target_col].replace({None: np.nan})

    def _fill(group):
        mode = group[target_col].mode()
        if not mode.empty:
            group[target_col] = group[target_col].fillna(mode.iat[0])
        return group


    df = df.groupby(group_col, group_keys=False).apply(_fill).reset_index(drop=True)
    return df

def fill_multiple_columns(df, group_col, target_cols):
    for col in target_cols:
        df = fill_missing_by_group(df, group_col, col)
    return df


def validate_sample_names(sample_df, column, mapping_dict):

    col_values = sample_df[column]
    errors = []
    validated_ids = []

    for uid, info in mapping_dict.items():
        resolved = info["resolved"]
        variations = info["variations"]

        # Count resolved
        resolved_count = (col_values == resolved).sum()

        # If resolved not present in sample, skip this uid entirely
        if resolved_count == 0:
            continue

        validated_ids.append(uid)

        # RULE 1: must appear exactly once
        if resolved_count != 1:
            errors.append(
                f"[ID {uid}] '{resolved}' appears {resolved_count} times in sample (should be once)."
            )

        # RULE 2: variations must NOT appear unless equivalent to resolved
        res_norm = resolved.lower()

        for var in variations:
            # Skip if variation is essentially same name
            if var == resolved or var.lower() == res_norm:
                continue

            # Now check if it appears
            if (col_values == var).any():
                errors.append(
                    f"[ID {uid}] Variation '{var}' found in sample while resolved is '{resolved}'."
                )

    # Output
    if errors:
        print("\n SAMPLE VALIDATION FAILED\n" + "\n".join(errors))
    else:
        print(f" Validation passed for all {len(validated_ids)} matching resolved names in the sample.")


def main():
    
    df=pd.read_csv('historical.csv')
    
    print("Data has been loaded into DF")
    
    df['first_name']=df['first_name'].apply(fix_strings)
    df['surname']=df['surname'].apply(fix_strings)
    df['birth_place']=df['birth_place'].apply(fix_strings)
    df['gender']=df['gender'].apply(fix_strings)
    df['occupation']=df['occupation'].apply(fix_strings)
    
    df['full_name'] = (df['first_name'].fillna('') + ' ' + df['surname'].fillna('')).str.strip()
    df['full_name'] = df['full_name'].replace('', None)
    
    print("Data has been cleaned for Clustering")
    
    full_name = df['full_name'].dropna().unique().tolist()
    distance_matrix_value=distance_matrix(full_name)
    cluster=hierarchical_cluster(distance_matrix_value,full_name)
    
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    resolved_names,ambiguous_names=name_recognized(ner_pipeline,cluster)
    filtered_auto_resolved,move_to_llm=low_confidence_name(resolved_names)
    
    print("Name has been recognized using Name Entity Recognization")
    
    ambiguous_names.update(move_to_llm)  
    
    llm_resolved_names=llm_component(ambiguous_names)
    
    print("Name has been recognized using LLM")
    
    final_name_list=llm_resolved_names | filtered_auto_resolved
    
    name_mapping = {}
    for cluster_id, data in final_name_list.items():
        resolved_name = data['resolved']
        for variation in data['variations']:
            name_mapping[variation] = resolved_name
            
    df['full_name'] = df['full_name'].map(name_mapping).fillna(df['full_name'])
    
    print("Names has been mapped to the recognized Names in the DF")
    
    filterd_df = fill_multiple_columns(df, group_col='full_name', target_cols=['birth_place','dob','gender', 'postcode', 'occupation'])
    
    filterd_df=filterd_df.drop_duplicates(subset=['full_name'], keep='first')
    
    filterd_df.to_csv('recreated_df.csv')
    
    print("Duplicate Name entries have been removed")
    
    sampled_df=filterd_df.sample(n=100,random_state=42)
    
    validate_sample_names(sampled_df, "full_name", final_name_list)
    
    print("Sample validation using Test Case has been conducted")

if __name__ == "__main__":
    main()