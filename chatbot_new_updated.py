from pathlib import Path
import os, json
import streamlit as st
# from getpass import getpass
from openai import OpenAI
import pandas as pd

# os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key: ")

client = OpenAI() 

SYSTEM_PROMPT = """You are a very experienced contract-review paralegal for commercial agreements.
Your task:
- For clauses in the uploaded contract, identify AT MOST FIVE most problematic clauses.
- Compare them against precedents where similar wordings in the precedent (past cases/contracts) caused disputes retrieved via File Search. Try your BEST to FIND precedents, even if they are just remotely related!!!
- For each risky clause, explicitly:
  1. Show the exact clause text (or short excerpt).
  2. Identify similar expressions in past contracts/cases retrieved, please LIST them EXPLICITLY! 
  3. Explain why those precedents led to disputes or legal issues.
  4. Explain why the current clause is problematic by analogy.
  5. Propose a redline (improved wording).
- CITE precedents inline as: CaseName (Year) - holding, with page/section if available.
- Quote sparingly (less than 300 words per precedent).
- It is always better to find some precedents versus none. However, if no relevant precedent is found, state clearly: “No supporting precedent found.”
- NEVER TRUNCATE YOUR OUTPUT!!!
"""
# - You are NOT a lawyer; this is educational analysis, not legal advice."""


def review_contract(file, vector_store_id: str, questions: str = ""):
    contract_file = client.files.create(file = file, purpose="user_data")

    print("We have successfully uploaded the file. Legal Contract Reviewer starts working!")

    resp = client.responses.create(
        model="o4-mini",
        # reasoning = {"effort": "high"},
        service_tier = "priority",
        truncation = "disabled",
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
            "max_num_results": 15
        }],
        include = ["file_search_call.results"],
        max_output_tokens = 50000,
        input=[
            {"role": 
                "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text", 
                        "text": (
                            "Task: Review the attached contract. Identify risky clauses, "
                            "propose redlines, and cite supporting cases from the knowledge base. "
                            # "Focus on indemnity, limitation of liability, termination, IP, governing law, and confidentiality. "
                            + (f"\nSpecial questions: {questions}" if questions else "")
                        ),
                    },
                    {
                        "type": "input_file", 
                        "file_id": contract_file.id
                    }
                ]
            }
        ]
    
    )

    try:
        print(resp.output_text)  # human-readable
    except Exception:
        pass

    # If using JSON schema, parse the structured object:
    try:
        data = resp.output[0].content[0].parsed  # ContractReview object
        return data
    except Exception:
        return {"raw": getattr(resp, "output_text", str(resp))}


# def split_paragraphs_merge_short(text, min_length=200):
#     paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() != ""]
#     chunks = []
#     buffer = ""
    
#     for para in paragraphs:
#         if len(buffer) > 0:
#             candidate = buffer + "\n\n" + para
#         else:
#             candidate = para
        
#         if len(candidate) < min_length:
#             buffer = candidate
#         else:
#             chunks.append(candidate)
#             buffer = ""
    
#     # Add any remaining buffer
#     if buffer:
#         chunks.append(buffer)
    
#     return chunks


eval_schema = {
    "type": "object",
    "properties": {
        "evaluation": {"type": "string"},
        "numeric_score": {"type": "number"}
    }, 
    "required":["evaluation", "numeric_score"]
}
    

def evaluate(eval_schema, inpt:str, *, system_context: str = "You are a contract law attorney, with high attention to detail for errors in contracts. It is important that you spot all errors, especially errors in legal contract reasoning.", model:str = "o4-mini"):
    gpt_response = client.chat.completions.create(
        model=model,
        service_tier = "priority",
        messages=[
            {"role": "system", "content": system_context},
            {"role": "user", "content": f"Your first-year paralegal has performed a legal contract analysis and it is now your job to evaluate their output. Their output has the following parts. 1: Finds a clause of problematic text in a legal contract. 2: Identifies similar expressions in past contracts/cases retrieved. 3: Explains why those precedents led to disputes or legal issues. 4: Explains why the current clause is problematic by analogy. 5: Proposes a improved wording. \n \n Your job is to evaluate the quality of their problematic wording analysis. \n \n Here is their analysis:\n\n {inpt}"}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Eval", "schema": eval_schema}
        }
    )
    return gpt_response.choices[0].message.content


# Your function that processes the document and returns a response
def process_document(contract_file, VECTOR_STORE_ID):
    result = review_contract(
        file = contract_file,
        vector_store_id = VECTOR_STORE_ID,
        questions= "",
    )
    return result

# Streamlit app
st.title("☂️ LegalChat")
st.subheader("A legal associate for everyone.")

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    st.write("File uploaded successfully! Analyzing...")
    response = process_document(contract_file = uploaded_file, VECTOR_STORE_ID = "vs_68a3b46135cc8191b2c7d3dfedffe471")
    st.subheader("Response:")
    st.write(response)

    st.write("Analysis successfully finished! Evaluating now...")

    output_path = "contract_review_temp.txt"
    with open(output_path, 'w') as f:
        print(response, file=f)
    
    with open(output_path, "r", encoding='ISO-8859-1') as f:
        text_str = f.read()

    # output_list = split_paragraphs_merge_short(text_str)
    # evaluation_df = pd.DataFrame(columns = ["evaluation", "numeric_score"])
    inp = response['raw']
    if type(response['raw'])!=str:
        print("response['raw'] not a str")

   
    evaluator_output = evaluate(eval_schema=eval_schema, inpt=f"The input, which may contain multiple critiques (each consisting of five elements, is as follows. For each critique, make sure to produce an evaluation and a score from 1-10. The input is: \n \n {inp}")
    eval_keys = json.loads(evaluator_output).keys()
    print(eval_keys)
    st.subheader("Evaluation Result:")
    for k in eval_keys:
        st.write(json.loads(evaluator_output)[k])

    # evl = json.loads(evaluator_output)["evaluation"]
    # scr = json.loads(evaluator_output)["numeric_score"]
    # evaluation_df.loc[len(evaluation_df)] = [evl, scr]
















