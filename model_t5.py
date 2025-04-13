import pdfplumber
import re
import json
import spacy
import os
from transformers import pipeline

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# Nuevos imports para IR con LangChain
# from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.docstore.document import Document

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings


##############################
# Cargar modelo NER en español
##############################
nlp = spacy.load("es_core_news_lg")  # Modelo grande en español para extracción de entidades

# Embedding models to compare
# embedding_models = [
#     "all-MiniLM-L6-v2",
#     "paraphrase-multilingual-MiniLM-L12-v2",
#     "msmarco-distilbert-base-v4",
#     "hiiamsid/sentence_similarity_spanish_es"
# ]
# embedding_models = ["paraphrase-multilingual-MiniLM-L12-v2"]
# embedding_models = ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
embedding_models = [
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    # "sentence-transformers/LaBSE",
    # "all-MiniLM-L6-v2",
    # "paraphrase-multilingual-MiniLM-L12-v2",
    # "msmarco-distilbert-base-v4",
    # "hiiamsid/sentence_similarity_spanish_es"
]

# chunk_sizes = [2500, 5000, 10000, 15000, "full_article"]
chunk_sizes = ["full_article"]

##############################
# Extracción del Texto del PDF
##############################
def extract_text_from_pdf(pdf_path):
    """
    Extrae todo el texto del PDF usando pdfplumber, página a página.
    """
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract plain text
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                table_text = "\n".join([" | ".join(row) for row in table if any(row)])  # Format table rows
                full_text += f"\n[Table Extracted]\n{table_text}\n"

    return full_text.strip()

##############################
# Preprocesamiento del Texto
##############################
def preprocess_text(text):
    """
    Limpia el texto eliminando múltiples saltos de línea y espacios redundantes.
    """
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

##############################
# Segmentación en Secciones por "Artículo"
##############################
def segment_text(text):
    """
    Divide el documento en secciones utilizando encabezados que empiezan con "Artículo".
    Retorna un diccionario donde la clave es el encabezado y el valor es el contenido hasta el siguiente encabezado.
    """
    parts = re.split(r'\n(Artículo\s+\d+\..*)', text)
    segments = {}
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        content = parts[i+1].strip() if (i+1) < len(parts) else ""
        segments[header] = content
    return segments



##############################
# NUEVAS FUNCIONES PARA IR CON LANGCHAIN
##############################
def load_and_split_pdf(pdf_path, chunk_size):
    """
    Carga el PDF completo, limpia el texto y lo divide en chunks según el chunk_size.
    """
    full_text = extract_text_from_pdf(pdf_path)
    full_text = preprocess_text(full_text)
    segments = segment_text(full_text)
    
    documents = []
    for article_header, article_content in segments.items():
        full_article_text = f"{article_header}\n{article_content}"
        if chunk_size == 'full_article':
            documents.append(Document(page_content=full_article_text, metadata={"source": pdf_path}))
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size * 0.1),
                add_start_index=True
            )
            sub_chunks = splitter.split_text(full_article_text)
            for sub_chunk in sub_chunks:
                documents.append(Document(page_content=sub_chunk, metadata={"source": pdf_path}))
    return documents


def create_vector_store(documents, model_name):
    """
    Crea un vector store utilizando embeddings de SentenceTransformer y Chroma.
    """
    embeddings = SentenceTransformerEmbeddings(model_name=model_name)
    # vector_store = Chroma.from_documents(documents, embeddings)
    collection_name = f"{model_name.split('/')[-1]}_dim{embeddings.client.get_sentence_embedding_dimension()}"
    return Chroma.from_documents(documents, embeddings, collection_name=collection_name)

    # return Chroma.from_documents(documents, embeddings, collection_name=f'{model_name.replace("/", "_")}_dim{embeddings.client.get_sentence_embedding_dimension()}')


##############################
# Extracción de Información Clave mediante IR y reglas
##############################

def extract_information_ir(vector_store, academic_year):
    """
    Extrae información clave del documento usando queries en lenguaje natural
    sobre el vector store.
    Se selecciona el artículo más relevante basado en la similitud semántica sin filtrar por palabras clave.
    """
    info = {"academic_year": academic_year}
    

    queries = {
        "funding": "Financiación de la convocatoria",
        "included_studies": "Enseñanzas comprendidas, enseñanzas postobligatorias y superiores no universitarias y universitarias del sistema educativo español con validez en todo el territorio nacional",
        "scholarship_types": "Clases y cuantías de las becas para cursar en el año académico, cuantías fijas y cuantía variable",
        "funding": "Financiación de la convocatoria",
        "beca_matricula": "Beneficiarios de beca de matrícula",
        "cuantía_fija": "Cuantía fija ligada a la renta del estudiante",
        "cuantia_fija_residencia": "Cuantía fija ligada a la residencia del estudiante durante el curso escolar",
        "cuantia_fija_excelencia_academica": "Cuantía fija ligada a la excelencia académica",
        "beca_basica": "Beneficiarios de beca básica",
        "scholarship_amounts": "Cuantías de las becas de carácter general",
        "general_requirements": "Requisitos generales para ser beneficiario de las becas",
        "income_thresholds": "Umbrales de renta familiar aplicables para la concesión de las becas",
        "academic_requirements": "Requisitos de carácter académico: normas comunes para todos los niveles para ser beneficiario de las becas",
        "credits": "Número de créditos de matrícula para obtener beca",
        "university_performance": "Rendimiento académico en el curso anterior para acceder a la universidad",
        "master_performance": "Rendimiento académico el curso anterior para acceder a estudios de máster",
        "beneficiary_obligations": "Obligaciones de los beneficiarios de las becas que se convocan",
        "application_documents": "Modelo de solicitud y documentación a presentar",
        "application_deadlines": "lugar y plazo de presentación de solicitudes",
        "review_and_corrections": "Revisión de solicitudes y subsanación de defectos y consulta del estado de tramitación",
        "notification_process": "Concesión, denegación, notificaciones y publicación",
        "compatibility_rules": "Compatibilidades de las becas e incompatibilidades con otras ayudas",
    }

    # Búsqueda de información
    for field, query in queries.items():
        results = vector_store.similarity_search(query, k=1)  # Buscar solo el mejor resultado
        selected_content = results[0].page_content.strip() if results else "No especificado"

        # Para montos, aplicamos regex para extraer valores exactos
        if field == "amounts":
            money_regex = r'(\d{1,3}(?:\.\d{3})*,\d{2}\s*(?:€|euros))'
            money_entities = re.findall(money_regex, selected_content, re.IGNORECASE)
            base_amount = money_entities[0] if len(money_entities) > 0 else "No especificado"
            additional_amount = money_entities[1] if len(money_entities) > 1 else "No especificado"
            info[field] = {"base_amount": base_amount, "additional_amount": additional_amount}
        else:
            info[field] = selected_content  # Guardamos el contenido extraído

    return info

##############################
# Función para Generar el Resumen a partir del JSON
##############################
def generate_summary_long(input_text, tokenizer, model, chunk_token_limit=768, max_length=256, min_length=100, num_passes=2):
    import torch
    from transformers import pipeline

    def summarize_chunks(text):
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=False)[0]
        total_tokens = len(input_ids)

        chunks = []
        for i in range(0, total_tokens, chunk_token_limit):
            chunk_ids = input_ids[i:i + chunk_token_limit]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)

        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        partial_summaries = []

        for idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            print(f"\nResumiendo chunk {idx + 1}/{len(chunks)}...")
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
            partial_summaries.append(summary)
            print(f"Resumen parcial {idx + 1}:\n{summary}\n")

        return "\n\n".join(partial_summaries)

    # Iteraciones de resumen
    current_text = input_text
    for pass_num in range(num_passes):
        print(f"\n🔁 Iteración de resumen {pass_num + 1}/{num_passes}...")
        current_text = summarize_chunks(current_text)

    return current_text



##############################
# Función Principal
##############################
def main():
    # Ruta al PDF
    pdf_path = "corpus_practical_assignment_1/ayudas_25-26.pdf"

    # Extraer el año académico del nombre del archivo (por ejemplo, "21-22" → "2021-2022")
    filename = os.path.basename(pdf_path)
    year_match = re.search(r'(\d{2})-(\d{2})', filename)
    if year_match:
        start_year = 2000 + int(year_match.group(1))
        end_year = 2000 + int(year_match.group(2))
        academic_year = f"{start_year}-{end_year}"
    else:
        academic_year = "Unknown"

    # splits = load_and_split_pdf(pdf_path)

    for chunk_size in chunk_sizes:
        for model in embedding_models:
            
            # # Guardamos la información extraída en un JSON
            # json_filename = f"scholarship_{academic_year}_{model.replace('/', '_')}_chunk_{chunk_size}.json"
            # print(f"Guardando información en {json_filename} ...")
            # with open(json_filename, 'w', encoding='utf-8') as f:
            #     json.dump(extracted_info, f, indent=4, ensure_ascii=False)
            #     print("Generando resumen a partir del JSON extraído...")
                
            from transformers import T5Tokenizer, MT5ForConditionalGeneration
            
            # Nombre del archivo JSON ya generado
            json_filename = f"scholarship_{academic_year}_sentence-transformers_paraphrase-multilingual-mpnet-base-v2_chunk_full_article.json"
            
            if os.path.exists(json_filename):
                print(f"Cargando información desde {json_filename} ...")
                with open(json_filename, 'r', encoding='utf-8') as f:
                    extracted_info = json.load(f)
            
                # Convertir la información extraída en un único bloque de texto para el resumen
                summary_input = " ".join([
                    f"{k}: {v}" if not isinstance(v, dict)
                    else f"{k}: {v.get('base_amount', 'No especificado')} / {v.get('additional_amount', 'No especificado')}"
                    for k, v in extracted_info.items()
                ])
            
                print("Cargando modelo y tokenizer...")
                model_name = "ELiRF/mt5-base-dacsa-es"
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = MT5ForConditionalGeneration.from_pretrained(model_name)
                
                print("Generando resumen por bloques...")
                generated_summary = generate_summary_long(summary_input, tokenizer, model, num_passes=2)
            
                print("\n--- Generated Summary ---\n")
                print(generated_summary)
                print("\n-------------------------\n")
                
                # Guardar resumen en archivo .txt
                output_filename = f"resumen_{academic_year}.txt"
                with open(output_filename, "w", encoding="utf-8") as out_file:
                    out_file.write(generated_summary)
                
                print(f"\n Resumen guardado en: {output_filename}")

            else:
                print(f"El archivo JSON {json_filename} no existe. Por favor, genera el JSON primero.")

    
if __name__ == '__main__':
    
    main()





