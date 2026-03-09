from pathlib import Path
import os
import re
import time
import datetime
import shutil
import tiktoken
import boto3
import httpx
import pandas as pd
import pprint
import json
import pyarrow as pa
import pyarrow.parquet as pq
from tabulate import tabulate
from dotenv import load_dotenv
import aioboto3
import instructor
import asyncio
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
import re
from lxml import etree
from xml.dom import minidom
from datetime import datetime
from bs4 import BeautifulSoup, Tag
from langchain_core.prompts import PromptTemplate
import os
import random
import string
import io
import gc
import traceback
import warnings
from contextlib import redirect_stdout
import copy
import fitz
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_model = None  

def get_embedding_model():
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def file_searcher(path):
    file_paths = []

    for file in path.rglob("*"):
        if file.is_file():
            file_paths.append(str(file))

    print("Total files found:", len(file_paths))
    return file_paths

def extract_text_from_pdf(pdf_paths):
    full_text = {}
    for i in pdf_paths:
        doc = fitz.open(i)
        full_text[i]=""
        for page in doc:
            full_text[i]+= page.get_text()

    return full_text

def chunk_by_file(file_dict):

    chunks = []

    for file_path, content in file_dict.items():
        chunks.append({
            "file_name": file_path,
            "content": content.strip()
        })

    return chunks

def create_vector_store(chunks):

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    model = get_embedding_model()
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks

def search(query, index, metadata, top_k):

    model = get_embedding_model()
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx, score in zip(indices[0], distances[0]):
        results.append({
            "file_name": metadata[idx]["file_name"],
            "content": metadata[idx]["content"],
            "score": float(score)
        })

    return results

def main_process(test_path):

    try:
        directory_path = Path(test_path)

        final_data_path = directory_path / "final_data"
        workings_path = directory_path / "workings"

        os.makedirs(final_data_path, exist_ok=True)
        os.makedirs(workings_path, exist_ok=True)

        path_list = file_searcher(directory_path)

        # Filter only PDFs
        path_list = [p for p in path_list if p.lower().endswith(".pdf")]

        if not path_list:
            return {
                "status": "error",
                "message": "No PDF files found in directory"
            }

        extracted_data = extract_text_from_pdf(path_list)
        chunks = chunk_by_file(extracted_data)
        index, metadata = create_vector_store(chunks)

        # ---- Find ITR file ----
        itr_search = search(
            "Indian Income Tax Return, ITR-1",
            index,
            metadata,
            len(path_list)
        )

        if not itr_search:
            return {
                "status": "error",
                "message": "Could not identify ITR file"
            }

        itr_file = itr_search[0]["file_name"]

        # ---- Get embedding of ITR file directly ----
        itr_index = next(
            i for i, m in enumerate(metadata)
            if m["file_name"] == itr_file
        )

        # FAISS internal vector
        itr_vector = index.reconstruct(itr_index)

        # ---- Search using ITR embedding ----
        distances, indices = index.search(
            np.array([itr_vector]).astype("float32"),
            len(path_list)
        )

        for idx, score in zip(indices[0], distances[0]):

            file_name = metadata[idx]["file_name"]

            # Skip the ITR file itself
            if file_name == itr_file:
                continue

            if score > 1.0:
                shutil.copy(file_name, workings_path)
            else:
                shutil.copy(file_name, final_data_path)

        return {
            "status": "success",
            "message": "Files Segregated successfully"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

