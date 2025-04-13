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
    "sentence-transformers/LaBSE",
    "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "msmarco-distilbert-base-v4",
    "hiiamsid/sentence_similarity_spanish_es"
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
# Extracción con NER
##############################
# def extract_entities(text):
#     """
#     Utiliza el modelo spaCy para extraer entidades de tipo MONEY, DATE y ORG.
#     Retorna un diccionario con listas de entidades encontradas para cada etiqueta.
#     """
#     doc = nlp(text)
#     entities = {"MONEY": [], "DATE": [], "ORG": []}
#     for ent in doc.ents:
#         if ent.label_ in entities:
#             entities[ent.label_].append(ent.text)
#     return entities

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
import re

def extract_information_ir(vector_store, academic_year):
    """
    Extrae información clave del documento usando queries en lenguaje natural
    sobre el vector store.
    Se selecciona el artículo más relevante basado en la similitud semántica sin filtrar por palabras clave.
    """
    info = {"academic_year": academic_year}
    
    # Old queries
    # queries = {
    #     "eligibility": "¿Cuáles son los requisitos específicos para ser beneficiario de la beca?",
    #     "educational_programs": "Enseñanzas comprendidas contempladas.",
    #     "amounts": "Indica el importe de las becas y las cuantías adicionales establecidas en la convocatoria.",
    #     "income_thresholds": "Lista los umbrales de renta familiares para la concesión de becas en la convocatoria.",
    #     "application_deadline": "Especifica la fecha límite para la presentación de solicitudes de becas.",
    #     "application_process": "Describe el procedimiento de solicitud de becas y la documentación requerida.",
    #     "general_requirements": "Lista los requisitos generales que deben cumplir los solicitantes de becas.",
    #     "academic_criteria": "Describe el rendimiento académico mínimo necesario para obtener la beca.",
    #     "incompatibilities": "Indica qué situaciones hacen que una beca no sea compatible con otras ayudas."
    # }

    # queries = {
    #     "funding": "¿Cuáles son los recursos financieros y cómo se asignan y gestionan según el tipo de beca?",
    #     "included_studies": "¿Qué enseñanzas están comprendidas en la convocatoria?",
    #     "scholarship_types": "¿Qué clases de becas están disponibles?",
    #     "scholarship_rent": "{Qué umbrales de renta se aplican según la clase de beca?",
    #     "scholarship_amounts": "¿Cuáles son las cuantías de las becas según su clase?",
    #     "scholarchip_benefits": "¿Quienes son los beneficiarios de cada clase de beca?",
    #     "scholarship_requirements": "¿Cuáles son los requisitos para solicitar cada clase de beca?",
    #     #   "tuition_scholarship": "¿En qué consiste la beca de matrícula?",
    #     #   "fixed_amount_income": "¿Cuál es la cuantía fija ligada a la renta del estudiante?",
    #     #   "fixed_amount_residence": "¿Cuál es la cuantía fija por residencia durante el curso escolar?",
    #     #   "fixed_amount_excellence": "¿Cuál es la cuantía fija por excelencia en el rendimiento académico?",
    #     #   "basic_scholarship": "¿Qué es la beca básica?",
    #     #   "additional_amounts": "¿Qué cuantías adicionales existen por domicilio insular o en Ceuta y Melilla?",
    #     #   "disability_scholarships": "¿Qué becas especiales existen para estudiantes con discapacidad?",
    #     #   "gender_violence_scholarships": "¿Cómo son las becas para víctimas de violencia de género o violencia sexual?",
    #     "general_requirements": "¿Cuáles son los requisitos generales para solicitar la beca?",
    #     "economic_requirements": "¿Qué requisitos económicos se exigen para la concesión de las becas?",
    #     #   "income_calculation": "¿Cómo se calcula la renta computable familiar?",
    #     #   "family_members": "¿Quiénes se consideran miembros computables para el cálculo de la renta?",
    #     #   "income_deductions": "¿Qué deducciones se aplican a la renta familiar?",
        
    #     "income_thresholds": "¿Cuáles son los umbrales de renta?",
    #     #   "wealth_thresholds": "¿Cuáles son los umbrales indicativos de patrimonio familiar?",
    #     #   "university_grade_average": "¿Cómo se calcula la nota media para estudios universitarios?",
    #     "academic_requirements": "¿Qué requisitos académicos se exigen para las becas?",
    #     "university_credits": "¿Cuál es el número de créditos de matrícula necesario para obtener la beca en estudios universitarios?",
    #     "university_performance": "¿Qué rendimiento académico previo se exige para las becas universitarias según rama o área de conocimiento?",
    #     #   "exceptional_performance": "¿Qué requisitos existen para el reconocimiento de excepcional rendimiento académico?",
    #     "double_degrees": "¿Cómo se regulan las becas en el caso de dobles titulaciones?",
    #     "master_credits": "¿Qué número de créditos de matrícula se exige en estudios de máster?",
    #     "master_performance": "¿Qué rendimiento académico se pide para estudios de máster?",
    #     "non_university_requirements": "¿Qué requisitos académicos aplican a las enseñanzas no universitarias?",
    #     "beneficiary_obligations": "¿Cuáles son las obligaciones de los beneficiarios de las becas?",
    #     "control_procedures": "¿Cómo se controlan y verifican las becas concedidas?",
    #     "application_documents": "¿Qué modelo de solicitúd y documentación se debe presentar y cómo?",
    #     "application_deadlines": "¿Cuál es el lugar y plazo de presentación de solicitudes?",
    #     "review_and_corrections": "¿Cómo se revisan las solicitudes y cuál es el plazo para corregir errores o presentar alegaciones?",
    #     "notification_process": "¿Cómo se notifica la concesión o denegación de la beca y dónde se publica?",
    #     "scholarship_payment": "¿Cómo se realiza el pago de las becas?",
    #     "compatibility_rules": "¿Qué compatibilidades tienen las becas con otras ayudas?"
    # }
    queries = {
        "funding": "Financiación de la convocatoria",
        "included_studies": "Enseñanzas comprendidas, enseñanzas postobligatorias y superiores no universitarias y universitarias del sistema educativo español con validez en todo el territorio nacional",
        "scholarship_types": "Clases y cuantías de las becas para cursar en el año académico, cuantías fijas y cuantía variable",
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
# Función Principal
##############################
def main():
    # Ruta al PDF
    pdf_path = "corpus_practical_assignment_1/ayudas_21-22.pdf"

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

    metrics = {}
    
    for chunk_size in chunk_sizes:
        for model in embedding_models:
            print(f"Evaluating {model} with chunk size {chunk_size}...")
            start_time = time.time()

            print("Cargando y dividiendo el PDF con LangChain...")
            splits = load_and_split_pdf(pdf_path, chunk_size)

            print(" Creando vector store...")
            vector_store = create_vector_store(splits, model)

            print("Extrayendo información clave mediante IR...")
            extracted_info = extract_information_ir(vector_store, academic_year)

            end_time = time.time()

            print("Información extraída:")
            
            retrieval_time = end_time - start_time
            
            metrics[model] = {
                "retrieval_time": retrieval_time,
            }

            # Guardamos la información extraída en un JSON
            json_filename = f"scholarship_{academic_year}_{model.replace('/', '_')}_chunk_{chunk_size}.json"
            print(f"Guardando información en {json_filename} ...")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(extracted_info, f, indent=4, ensure_ascii=False)

    
if __name__ == '__main__':
    
    main()





