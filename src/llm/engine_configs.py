from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import VertexAI
from google.oauth2 import service_account
from google.cloud import aiplatform
from typing import Union, List, Dict, Any
import vertexai
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import os
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.prompt_values import ChatPromptValue
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import sys
import re
sys.path.append(".")
    
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}


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

"""
This module defines configurations for various language models using the langchain library.
Each configuration includes a constructor, parameters, and an optional preprocessing function.
"""

model_path='/fred/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/74fbf131a939963dd1e244389bb61ad0d0440a4d'
cuda_visible='0,1,2,3,4,5,6,7'
os.environ["HF_DATASETS_CACHE"] = model_path
os.environ["HF_HOME"] = model_path
os.environ["HF_HUB_CACHE"] = model_path
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible

class CustomHuggingFacePipeline(HuggingFacePipeline):
    def invoke(self, input: Union[str, List[Union[List[str], tuple[str, str], str, Dict[str, Any]]]], config = None, *, stop = None, **kwargs) -> AIMessage: 
        # Handle stop words
        generation_kwargs = {}
        if stop:
            generation_kwargs["stop_sequences"] = stop

        # Handle different input types
        prompt = self._format_input(input)
        # generated_text = self.pipeline(prompt, **kwargs)[0]["generated_text"]
        # Generate text with stop words
        generated_text = self.pipeline(prompt, **{**kwargs, **generation_kwargs})[0]["generated_text"]

        cleaned = re.sub(r"<(thought|thoughts|system)>.*?</\1>", "", generated_text, flags=re.DOTALL).strip()
        if '</think>' in cleaned:
            parts = cleaned.split('</think>', 1)
            response = parts[1].strip()
        else:
            response = cleaned.strip()
        return AIMessage(content=response)
    
    def _format_input(
        self, 
        input: Union[
            str, 
            List[Union[
                    List[str], 
                    tuple[str, str], 
                    str, 
                    Dict[str, Any]
                ]
            ]
        ]
    ) -> str:
        """Converts various input types into a unified prompt string."""
        if isinstance(input, str):
            return input
        # Add handling for ChatPromptValue
        elif isinstance(input, ChatPromptValue):
            return input.to_string()
        elif isinstance(input, list) and not isinstance(input, str):
            messages = []
            for item in input:
                # If item is a string, include as-is.
                if isinstance(item, str):
                    messages.append(item)
                # Handle objects with attributes role and content (e.g. BaseMessage)
                elif hasattr(item, "role") and hasattr(item, "content"):
                    messages.append(f"{item.role}: {item.content}")
                # Handle dictionaries.
                elif isinstance(item, dict):
                    # If the dict has 'role' and 'content', use them.
                    if "role" in item and "content" in item:
                        messages.append(f"{item['role']}: {item['content']}")
                    else:
                        # Otherwise, join all key-value pairs.
                        formatted_pairs = [f"{k}: {v}" for k, v in item.items()]
                        messages.append("\n".join(formatted_pairs))
                # Handle lists and tuples.
                elif isinstance(item, (list, tuple)):
                    if len(item) == 2 and all(isinstance(x, str) for x in item):
                        messages.append(f"{item[0]}: {item[1]}")
                    else:
                        messages.append(" ".join(str(x) for x in item))
                else:
                    messages.append(str(item))
            return "\n".join(messages)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

def create_local_model(model_path, temperature=0.1):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=5000,
        temperature=temperature,
        top_p=0.9,
    )
    return CustomHuggingFacePipeline(pipeline=hf_pipeline)

# class LocalModel:
#     def __init__(self, model_path, temperature=0.2):
#         self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#         self.default_temperature = temperature

#     def invoke(self, 
#               input: Union[str, List[Dict[str, str]]], 
#               config: None = None, 
#               *, 
#               stop: List[str] = None, 
#               **kwargs) -> AIMessage:
#         # Handle different input types
#         prompt = self._format_input(input)
        
#         # Get generation parameters
#         temperature = kwargs.get('temperature', self.default_temperature)

#         # Tokenize and generate
#         inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
#         outputs = self.model.generate(
#             inputs.input_ids,
#             max_length=3000,
#             temperature=temperature,
#             do_sample=True,
#             pad_token_id=self.tokenizer.eos_token_id,
#             stopping_criteria=stop  # Add stop words handling if needed
#         )

#         # Decode and calculate tokens
#         generated_text = self.tokenizer.decode(
#             outputs[0][inputs.input_ids.shape[-1]:], 
#             skip_special_tokens=True
#         )
        
#         return self._create_ai_message(prompt, generated_text)

#     def _format_input(self, input: Union[str, List]) -> str:
#         """Handle different input types"""
#         if isinstance(input, str):
#             return input
#         elif isinstance(input, list):
#             return "\n".join([f"{m['role']}: {m['content']}" for m in input])
#         else:
#             raise ValueError(f"Unsupported input type: {type(input)}")

#     def _create_ai_message(self, prompt: str, response: str) -> AIMessage:
#         """Create AIMessage with metadata"""
#         prompt_tokens = len(self.tokenizer.encode(prompt))
#         completion_tokens = len(self.tokenizer.encode(response))
        
#         return AIMessage(
#             content=response.strip(),
#             response_metadata={
#                 'token_usage': {
#                     'prompt_tokens': prompt_tokens,
#                     'completion_tokens': completion_tokens,
#                     'total_tokens': prompt_tokens + completion_tokens
#                 },
#                 'model_name': 'local-model',
#                 'system_fingerprint': 'fp_custom',
#                 'finish_reason': 'stop'
#             }
#         )


ENGINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "r1-distill": {
        "constructor": create_local_model,
        "params": {"model_path": model_path, "temperature": 0.1},
    },
    "gemini-pro": {
        "constructor": ChatGoogleGenerativeAI,
        "params": {"model": "gemini-pro", "temperature": 0},
        "preprocess": lambda x: x.to_messages()
    },
    "gemini-1.5-pro": {
        "constructor": VertexAI,
        "params": {"model": "gemini-1.5-pro", "temperature": 0, "safety_settings": safety_settings}
    },
    "gemini-1.5-pro-002": {
        "constructor": VertexAI,
        "params": {"model": "gemini-1.5-pro-002", "temperature": 0, "safety_settings": safety_settings}
    },
    "gemini-1.5-flash":{
        "constructor": VertexAI,
        "params": {"model": "gemini-1.5-flash", "temperature": 0, "safety_settings": safety_settings}
    },
    "picker_gemini_model": {
        "constructor": VertexAI,
        "params": {"model": "projects/613565144741/locations/us-central1/endpoints/7618015791069265920", "temperature": 0, "safety_settings": safety_settings}
    },
    "gemini-1.5-pro-text2sql": {
        "constructor": VertexAI,
        "params": {"model": "projects/618488765595/locations/us-central1/endpoints/1743594544210903040", "temperature": 0, "safety_settings": safety_settings}
    },
    "cot_picker": {
        "constructor": VertexAI,
        "params": {"model": "projects/243839366443/locations/us-central1/endpoints/2772315215344173056", "temperature": 0, "safety_settings": safety_settings}
    },
    "gpt-3.5-turbo-0125": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-3.5-turbo-0125", "temperature": 0}
    },
    "gpt-3.5-turbo-instruct": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-3.5-turbo-instruct", "temperature": 0}
    },
    "gpt-4-1106-preview": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4-1106-preview", "temperature": 0}
    },
    "gpt-4-0125-preview": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4-0125-preview", "temperature": 0}
    },
    "gpt-4-turbo": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4-turbo", "temperature": 0}
    },
    "gpt-4o": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4o", "temperature": 0}
    },
    "gpt-4o-mini": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4o-mini", "temperature": 0}
    },
    "claude-3-opus-20240229": {
        "constructor": ChatAnthropic,
        "params": {"model": "claude-3-opus-20240229", "temperature": 0}
    },
    # "finetuned_nl2sql": {
    #     "constructor": ChatOpenAI,
    #     "params": {
    #         "model": "AI4DS/NL2SQL_DeepSeek_33B",
    #         "openai_api_key": "EMPTY",
    #         "openai_api_base": "/v1",
    #         "max_tokens": 400,
    #         "temperature": 0,
    #         "stop": ["```\n", ";"]
    #     }
    # },
    "finetuned_nl2sql": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "ft:gpt-4o-mini-2024-07-18:stanford-university::9p4f6Z4W",
            "max_tokens": 400,
            "temperature": 0,
            "stop": ["```\n", ";"]
        }
    },
    "column_selection_finetuning": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "ft:gpt-4o-mini-2024-07-18:stanford-university::9t1Gcj6Y:ckpt-step-1511",
            "max_tokens": 1000,
            "temperature": 0,
            "stop": [";"]
        }
    },
    # "finetuned_nl2sql_cot": {
    #     "constructor": ChatOpenAI,
    #     "params": {
    #         "model": "AI4DS/deepseek-cot",
    #         "openai_api_key": "EMPTY",
    #         "openai_api_base": "/v1",
    #         "max_tokens": 1000,
    #         "temperature": 0,
    #         "stop": [";"]
    #     }
    # },
    # "finetuned_nl2sql_cot": {
    #     "constructor": ChatOpenAI,
    #     "params": {
    #         "model": "ft:gpt-4o-mini-2024-07-18:stanford-university::9oKvRYet",
    #         "max_tokens": 1000,
    #         "temperature": 0,
    #         "stop": [";"]
    #     }
    # },
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "meta-llama/Meta-Llama-3-70B-Instruct",
            "openai_api_key": "EMPTY",
            "openai_api_base": "/v1",
            "max_tokens": 600,
            "temperature": 0,
            "model_kwargs": {
                "stop": [""]
            }
        }
    }
}
