import streamlit as st

from transformers import AutoTokenizer, AutoModel
import torch 
import pandas as pd 
import numpy as np 
import torch.nn.functional as F
import itertools
import json 
import chardet
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
from openai import OpenAI
from docx import Document
from legal_bert_embed import * 

data_path = "/Users/carolgao/MIT Dropbox/Carol Gao/common_experience_2025/"
client = OpenAI()


# Your function that processes the document and returns a response
def process_document(file):
    text = uploaded_file.read().decode("ISO-8859-1")
    chunks = split_paragraphs_merge_short(text)

    df_contract = pd.DataFrame()
    for i in range(0, len(chunks)):
        embedding = get_embedding(chunks[i])
        df_temp = pd.DataFrame([embedding])
        df_temp['chunk'] = i 
        df_contract = pd.concat([df_contract, df_temp])
        print("finished chunk", i)
    df_contract = df_contract.reset_index(drop =True)

    df_ruling = pd.read_csv(f"{data_path}/rulings/case_embeddings.csv")
    df_contract_matrix = df_contract.drop(columns = {"chunk"})
    df_ruling_matrix = df_ruling.drop(columns = {"case", "flaw_index"})
    pairs = list(itertools.product(df_contract_matrix.index, df_ruling_matrix.index))
    cos_sim_matrix = pd.DataFrame(cosine_similarity(df_contract_matrix, df_ruling_matrix).flatten()
                    , index=pd.MultiIndex.from_tuples(pairs, names=['contract_index', 'ruling_index'])).rename(columns = {0: "cos_sim"})
    top_indices = cos_sim_matrix.sort_values(by = 'cos_sim', ascending = False).head(5).index
    top_cases = pd.DataFrame(columns = ["contract_chunk", "case", "case_flaw_index"])
    for i in range(0, len(top_indices)):
        contract_chunk = df_contract.loc[df_contract.index == top_indices[i][0]]['chunk']
        case = df_ruling.loc[df_ruling.index == top_indices[i][1]]['case']
        case_flaw = df_ruling.loc[df_ruling.index == top_indices[i][1]]['flaw_index']
        top_cases = pd.concat([top_cases, pd.DataFrame([np.array([contract_chunk, case, case_flaw]).flatten()], columns = ["contract_chunk", "case", "case_flaw_index"])])
    top_cases['contract'] = ""
    for i in top_cases['contract_chunk'].unique():
        top_cases.loc[top_cases['contract_chunk'] == i, 'contract'] = chunks[i]
    top_cases.to_csv(f"{data_path}/contracts/test.csv", index = False)
    
    df = pd.read_csv(f"{data_path}/contracts/test.csv")
    df_reasoning = pd.DataFrame() 
    for i in range(0, 1):
        contract_text = df['contract'][i]
        case_name = str(df['case'][i])
        print(str(case_name))
        case_flaw_index = df['case_flaw_index'][i]
        with open(f"{data_path}/rulings/Trial_678_rulings_output/{case_name}", "r") as f:
            data = json.load(f)
        case_text = str(data['flaws'][case_flaw_index])

        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are a legal expert."},
                {"role": "user", "content": f"This is a segment of a new contract: {contract_text} \n  \
                This is the summary of a ruling from a previous similar case: {case_text} \n \
                Can you summarize how this new contract might be subject to similar flaws or disputes as identified by the previous ruling?"}
            ],
        )
    df_reasoning = pd.concat([df_reasoning, pd.DataFrame([response.choices[0].message.content])])

    response = f"{df_reasoning[0][0]}"
    return response



# Streamlit app
st.title("☂️ LegalChat")
st.subheader("An associate for legal assocaites.")

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    st.write("File uploaded successfully! Analyzing...")
    response = process_document(uploaded_file)
    st.subheader("Response:")
    st.write(response)
