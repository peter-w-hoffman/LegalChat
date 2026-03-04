from pathlib import Path
import os, json
from getpass import getpass
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key: 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")
client = OpenAI()

stores = client.vector_stores.list()
for vs in stores.data:
    print(vs.id, vs.name, getattr(vs, "file_counts", getattr(vs, "fileCounts", None)))