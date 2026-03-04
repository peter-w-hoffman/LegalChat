
from transformers import AutoTokenizer, AutoModel
import torch 
import pandas as pd 
import numpy as np 
import torch.nn.functional as F
import json 
import os
import sys
from legal_bert_embed import *

CONTRACT = "2177CV00402"

data_path = "/Users/carolgao/MIT Dropbox/Carol Gao/common_experience_2025/"

with open(f"{data_path}/contracts/corrected_output_{CONTRACT}.txt", "r", encoding='ISO-8859-1') as f:
    text_str = f.read()


chunks = split_paragraphs_merge_short(text_str)

df = pd.DataFrame()
for i in range(0, len(chunks)):
    embedding = get_embedding(chunks[i])
    df_temp = pd.DataFrame([embedding])
    df_temp['chunk'] = i 
    df = pd.concat([df, df_temp])
    print("finished chunk", i)

df.to_csv(f"{data_path}/contracts/{CONTRACT}_embeddings.csv", index = False)