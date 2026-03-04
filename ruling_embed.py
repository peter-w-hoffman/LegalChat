
from transformers import AutoTokenizer, AutoModel
import torch 
import pandas as pd 
import numpy as np 
import torch.nn.functional as F
import json 
import os
from legal_bert_embed import *

data_path = "/Users/carolgao/MIT Dropbox/Carol Gao/common_experience_2025/"


df_embed = pd.DataFrame()
folder = f"{data_path}/rulings/Trial_678_rulings_output"
for filename in os.listdir(folder):
    try:
        if filename.endswith(".json"):
            file_path = os.path.join(folder, filename)
            with open(file_path, "r") as f:
                data = json.load(f)
            for i in range(0,3):
                text = data['flaws'][i]['original_expressions'][0]
                embedding = get_embedding(text)
                df_temp = pd.DataFrame([embedding])
                df_temp['case'] = filename
                df_temp['flaw_index'] = i
                df_embed = pd.concat([df_embed, df_temp])
                print(len(text))                
    except:
        print(filename, "empty: SKIPPED")

df_embed.to_csv(f"{data_path}/rulings/case_embeddings.csv", index = False)