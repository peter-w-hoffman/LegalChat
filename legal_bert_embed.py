
from transformers import AutoTokenizer, AutoModel
import torch 
import pandas as pd 
import numpy as np 
import torch.nn.functional as F
import json 
import os

# load legal bert 
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cpu")
model.to(device)

# tokenize
def tokenize_and_reshape(text, max_length=512):
    # Tokenize texts without splitting into chunks first
    encoded = tokenizer(text, return_tensors="pt")
    # Determine padding necessary to make input_ids length a multiple of max_length
    current_length = encoded.input_ids.size(1)
    padding_needed = (max_length - current_length % max_length) % max_length
    # Apply padding using torch.nn.functional.pad
    pad_config = (0, padding_needed)
    padded_input_ids = F.pad(encoded.input_ids, pad_config, value=tokenizer.pad_token_id)
    padded_token_type_ids = F.pad(encoded.token_type_ids, pad_config, value=0)
    padded_attention_mask = F.pad(encoded.attention_mask, pad_config, value=0)
    # Reshape the tensors to have sequences of max_length and send to cuda
    chunks = {
        'input_ids': padded_input_ids.reshape(-1, max_length).to(device),
        'token_type_ids': padded_token_type_ids.reshape(-1, max_length).to(device),
        'attention_mask': padded_attention_mask.reshape(-1, max_length).to(device)
    }
    return chunks

# get embeddings
def get_embedding(text):
    model.eval()
    with torch.no_grad():
        tokens = tokenize_and_reshape(text)

        # Obtain embeddings from model
        output = model(**tokens).pooler_output

        note_embedding = output.mean(dim=0).detach().to('cpu').numpy()

    torch.cuda.empty_cache() 

    return note_embedding


def split_paragraphs_merge_short(text, min_length=1000):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() != ""]
    chunks = []
    buffer = ""
    
    for para in paragraphs:
        if len(buffer) > 0:
            candidate = buffer + "\n\n" + para
        else:
            candidate = para
        
        if len(candidate) < min_length:
            buffer = candidate
        else:
            chunks.append(candidate)
            buffer = ""
    
    # Add any remaining buffer
    if buffer:
        chunks.append(buffer)
    
    return chunks
