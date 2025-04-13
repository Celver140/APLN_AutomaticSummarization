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

embedding_models = [
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]
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
def generate_summary_long(input_text, tokenizer, model, chunk_token_limit=1024, final_max_tokens=350):
    import torch
    from transformers import pipeline

    def split_into_chunks(text, limit):
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        chunks = []
        for i in range(0, len(tokens), limit):
            chunk_ids = tokens[i:i+limit]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
        return chunks

    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

    print("\n Primera pasada: resumen por chunks largos...")
    long_chunks = split_into_chunks(input_text, chunk_token_limit)
    partial_summaries = []
    for idx, chunk in enumerate(long_chunks):
        if len(chunk.strip()) < 50:
            continue
        print(f"Resumiendo chunk {idx+1}/{len(long_chunks)}...")
        try:
            summary = summarizer(chunk, max_length=256, min_length=80, truncation=True)[0]["summary_text"]
            partial_summaries.append(summary)
        except Exception as e:
            print(f"Error en chunk {idx+1}: {e}")

    if not partial_summaries:
        return "No se pudo generar resumen."

    all_summary_text = " ".join(partial_summaries)
    token_count = len(tokenizer(all_summary_text)["input_ids"])

    if token_count <= final_max_tokens:
        print("\nEl resumen ya es suficientemente corto.")
        return all_summary_text

    print("\nSegunda pasada: resumen final...")

    try:
        second_chunks = split_into_chunks(all_summary_text, chunk_token_limit)
        second_pass = []
        for idx, chunk in enumerate(second_chunks):
            print(f"Refinando resumen {idx+1}/{len(second_chunks)}...")
            refined = summarizer(chunk, max_length=final_max_tokens, min_length=150, truncation=True)[0]["summary_text"]
            second_pass.append(refined)

        final_text = " ".join(second_pass)
        final_token_count = len(tokenizer(final_text)["input_ids"])
        if final_token_count > final_max_tokens:
            # Último truncamiento
            print("Truncando resumen final...")
            final_tokens = tokenizer(final_text, return_tensors="pt")["input_ids"][0][:final_max_tokens]
            final_text = tokenizer.decode(final_tokens, skip_special_tokens=True)

        return final_text

    except Exception as e:
        print(f"Error generando resumen final: {e}")
        return all_summary_text  # fallback

def clean_text(text):
    text = re.sub(r'https?://\S+', '', text)  # eliminar URLs
    text = re.sub(r'IQUEN[A-Z]*', '', text)   # eliminar tags corruptos
    text = re.sub(r'VALIDACIÓN.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'FIRMANTE.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[A-Z0-9]{8,}\b', '', text)  # eliminar códigos largos
    text = re.sub(r'\s{2,}', ' ', text)  # eliminar espacios duplicados
    return text.strip()




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

    for chunk_size in chunk_sizes:
        for model in embedding_models:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

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

                summary_input= clean_text(summary_input)  # Limpiar el texto antes de resumir
            
                print("Cargando modelo y tokenizer...")
                model_name = "mrm8488/bart-legal-base-es"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

                print("Generando resumen por bloques...")
                generated_summary = generate_summary_long(summary_input, tokenizer, model) #, num_passes=2)
            
                print("\n--- Generated Summary ---\n")
                print(generated_summary)
                print("\n-------------------------\n")
                
                # Guardar resumen en archivo .txt
                output_filename = f"resumen_{academic_year}_{model_name.split('/')[-1]}.txt"
                with open(output_filename, "w", encoding="utf-8") as out_file:
                    out_file.write(generated_summary)
                
                print(f"\n Resumen guardado en: {output_filename}")

            else:
                print(f"El archivo JSON {json_filename} no existe. Por favor, genera el JSON primero.")

    
if __name__ == '__main__':
    
    main()





