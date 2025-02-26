import os
from pathlib import Path
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema.document import Document
# from langchain_openai import OpenAIEmbeddings
# from langchain_google_vertexai import VertexAIEmbeddings
from transformers import AutoModel, AutoTokenizer
import torch
from google.oauth2 import service_account
from google.cloud import aiplatform
import vertexai
import sys
sys.path.append(".")

from database_utils.db_catalog.csv_utils import load_tables_description

load_dotenv(override=True)

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_REGION = os.getenv("GCP_REGION")
GCP_CREDENTIALS = os.getenv("GCP_CREDENTIALS")

if GCP_CREDENTIALS and GCP_PROJECT and GCP_REGION:
    aiplatform.init(
    project=GCP_PROJECT,
    location=GCP_REGION,
    credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS)
    )
    vertexai.init(project=GCP_PROJECT, location=GCP_REGION, credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS))


# EMBEDDING_FUNCTION = VertexAIEmbeddings(model_name="text-embedding-004")#OpenAIEmbeddings(model="text-embedding-3-large")
# EMBEDDING_FUNCTION = OpenAIEmbeddings(model="text-embedding-3-large")
model_path='../models/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693'
cuda_visible='0,1,2,3'
os.environ["HF_DATASETS_CACHE"] = model_path
os.environ["HF_HOME"] = model_path
os.environ["HF_HUB_CACHE"] = model_path
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible

class LocalEmbeddings:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            output_hidden_states=True
        )

    def embed_documents(self, texts):
        # Tokenize and move inputs to the same device as the model
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use attention_mask to ignore padding tokens in mean pooling
        attention_mask = inputs.attention_mask.unsqueeze(-1)
        last_hidden_state = outputs.last_hidden_state
        embeddings = (last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        return embeddings.to(torch.float32).cpu().numpy()
        # return outputs.last_hidden_state.mean(dim=1).to(torch.float32).cpu().numpy()
    
    def embed_query(self, text):
        embeddings = self.embed_documents([text])
        return embeddings[0]

EMBEDDING_FUNCTION = LocalEmbeddings(model_path=model_path)



def make_db_context_vec_db(db_directory_path: str, **kwargs) -> None:
    """
    Creates a context vector database for the specified database directory.

    Args:
        db_directory_path (str): The path to the database directory.
        **kwargs: Additional keyword arguments, including:
            - use_value_description (bool): Whether to include value descriptions (default is True).
    """
    db_id = Path(db_directory_path).name

    table_description = load_tables_description(db_directory_path, kwargs.get("use_value_description", True))
    docs = []
    
    for table_name, columns in table_description.items():
        for column_name, column_info in columns.items():
            metadata = {
                "table_name": table_name,
                "original_column_name": column_name,
                "column_name": column_info.get('column_name', ''),
                "column_description": column_info.get('column_description', ''),
                "value_description": column_info.get('value_description', '') if kwargs.get("use_value_description", True) else ""
            }
            for key in ['column_name', 'column_description', 'value_description']:
                if column_info.get(key, '').strip():
                    docs.append(Document(page_content=column_info[key], metadata=metadata))
    
    logging.info(f"Creating context vector database for {db_id}")
    vector_db_path = Path(db_directory_path) / "context_vector_db"

    if vector_db_path.exists():
        os.system(f"rm -r {vector_db_path}")

    vector_db_path.mkdir(exist_ok=True)

    Chroma.from_documents(docs, EMBEDDING_FUNCTION, persist_directory=str(vector_db_path))

    logging.info(f"Context vector database created at {vector_db_path}")
