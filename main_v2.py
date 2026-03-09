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

import re

def normalize_money(value):
    return float(value.replace(",", "").strip())

def normalize_text(value):
    return value.strip().upper()

def extract_itr_fields(text):
    data = {}

    # Total Income
    match = re.search(r"Total Income\s*[:\-]?\s*([\d,]+)", text, re.IGNORECASE)
    if match:
        data["total_income"] = normalize_money(match.group(1))

    # Tax Paid
    match = re.search(r"Tax Paid\s*[:\-]?\s*([\d,]+)", text, re.IGNORECASE)
    if match:
        data["tax_paid"] = normalize_money(match.group(1))

    # Assessment Year
    match = re.search(r"Assessment Year\s*[:\-]?\s*([\d\-]+)", text, re.IGNORECASE)
    if match:
        data["assessment_year"] = normalize_text(match.group(1))

    # PAN
    match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text)
    if match:
        data["pan"] = normalize_text(match.group(0))

    return data

def calculate_match_score(itr_data, candidate_data):

    if not itr_data:
        return 0

    matches = 0
    total_fields = len(itr_data)

    for field, itr_value in itr_data.items():
        candidate_value = candidate_data.get(field)

        if candidate_value is None:
            continue

        # Numeric tolerance
        if isinstance(itr_value, float):
            if abs(itr_value - candidate_value) <= 1:
                matches += 1
        else:
            if itr_value == candidate_value:
                matches += 1

    return matches / total_fields

def main_process(test_path):

    try:
        directory_path = Path(test_path)

        final_data_path = directory_path / "final_data"
        workings_path = directory_path / "workings"

        os.makedirs(final_data_path, exist_ok=True)
        os.makedirs(workings_path, exist_ok=True)

        path_list = file_searcher(directory_path)
        path_list = [p for p in path_list if p.lower().endswith(".pdf")]

        if not path_list:
            return {
                "status": "error",
                "message": "No PDF files found"
            }

        # Extract text from all PDFs
        extracted_data = extract_text_from_pdf(path_list)

        # Identify ITR document via keyword presence
        itr_file = None
        for file_path, content in extracted_data.items():
            if "income tax return" in content.lower():
                itr_file = file_path
                break
  
        if not itr_file:
            return {
                "status": "error",
                "message": "ITR document not identified"
            }

        itr_data = extract_itr_fields(extracted_data[itr_file])

        if not itr_data:
            return {
                "status": "error",
                "message": "Failed to extract fields from ITR"
            }

        # Compare other documents
        for file_path, content in extracted_data.items():

            if file_path == itr_file:
                continue

            candidate_data = extract_itr_fields(content)

            score = calculate_match_score(itr_data, candidate_data)

            if score >= 0.75:
                shutil.copy(file_path, workings_path)
            else:
                shutil.copy(file_path, final_data_path)

        return {
            "status": "success",
            "message": "Files segregated based on numeric validation"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

