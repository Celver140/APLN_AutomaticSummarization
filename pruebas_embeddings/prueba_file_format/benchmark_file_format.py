import pdfplumber
import re
import json
import spacy
import os
import time
import numpy as np
import pandas as pd
import h5py
import xml.etree.ElementTree as ET
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# Benchmark different formats for storing PDF extracted text
def benchmark_storage_formats(pdf_text, base_filename="pdf_extracted_data"):
    results = {}

    # JSON
    start = time.time()
    json_filename = f"{base_filename}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump({"content": pdf_text}, f, ensure_ascii=False, indent=4)
    json_time = time.time() - start
    json_size = os.path.getsize(json_filename)

    # CSV (line by line text entries)
    start = time.time()
    csv_filename = f"{base_filename}.csv"
    df = pd.DataFrame({"text": pdf_text.split('\n')})
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    csv_time = time.time() - start
    csv_size = os.path.getsize(csv_filename)

    # XML
    start = time.time()
    xml_filename = f"{base_filename}.xml"
    root = ET.Element("document")
    for line in pdf_text.split('\n'):
        line_el = ET.SubElement(root, "line")
        line_el.text = line
    tree = ET.ElementTree(root)
    tree.write(xml_filename, encoding='utf-8', xml_declaration=True)
    xml_time = time.time() - start
    xml_size = os.path.getsize(xml_filename)

    # HDF5
    start = time.time()
    hdf5_filename = f"{base_filename}.h5"
    with h5py.File(hdf5_filename, 'w') as hf:
        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset("lines", data=pdf_text.split('\n'), dtype=dt)
    hdf5_time = time.time() - start
    hdf5_size = os.path.getsize(hdf5_filename)

    # Collect results
    results["JSON"] = {"size_bytes": json_size, "write_time_sec": json_time}
    results["CSV"] = {"size_bytes": csv_size, "write_time_sec": csv_time}
    results["XML"] = {"size_bytes": xml_size, "write_time_sec": xml_time}
    results["HDF5"] = {"size_bytes": hdf5_size, "write_time_sec": hdf5_time}

    print(json.dumps(results, indent=4))
    return results


if __name__ == '__main__':
    pdf_path = "corpus_practical_assignment_1/ayudas_21-22.pdf"
    with pdfplumber.open(pdf_path) as pdf:
        pdf_text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text + "\n"
            tables = page.extract_tables()
            for table in tables:
                table_text = "\n".join([" | ".join(row) for row in table if any(row)])
                pdf_text += f"\n[Table Extracted]\n{table_text}\n"

    benchmark_storage_formats(pdf_text, base_filename="pdf_storage_comparison")
