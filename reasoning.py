import pandas as pd 
import numpy as np 
import json 
import os
from openai import OpenAI
from docx import Document

client = OpenAI()

CONTRACT = "2177CV00402"
data_path = "/Users/carolgao/MIT Dropbox/Carol Gao/common_experience_2025/"

df = pd.read_csv(f"{data_path}/contracts/top_pairs_{CONTRACT}_new.csv")

df_reasoning = pd.DataFrame()
for i in range(0, len(df)):
    contract_text = df['contract'][i]
    case_name = df['case'][i]
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

df_reasoning.to_csv(f"{data_path}/contracts/reasoning_{CONTRACT}_new.csv", index = False)
    

    

