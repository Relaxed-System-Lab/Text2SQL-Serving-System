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
import torch
sys.path.append(".")
    
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}


# GCP_PROJECT = os.getenv("GCP_PROJECT")
# GCP_REGION = os.getenv("GCP_REGION")
# GCP_CREDENTIALS = os.getenv("GCP_CREDENTIALS")

# if GCP_CREDENTIALS and GCP_PROJECT and GCP_REGION:
#     aiplatform.init(
#     project=GCP_PROJECT,
#     location=GCP_REGION,
#     credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS)
#     )
#     vertexai.init(project=GCP_PROJECT, location=GCP_REGION, credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS))

"""
This module defines configurations for various language models using the langchain library.
Each configuration includes a constructor, parameters, and an optional preprocessing function.
"""

model_path1='../models/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693'
model_path2='../models/models--tablegpt--TableGPT2-7B/snapshots/9de1c2116151f6ccc6915616f625bb9c365dd9ba'
model_path3='../models/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b'
cuda_visible='0,1,2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible

class CustomHuggingFacePipeline(HuggingFacePipeline):
    def invoke(self, input: Union[str, List[Union[List[str], tuple[str, str], str, Dict[str, Any]]]], config = None, *, stop = None, **kwargs) -> AIMessage: 
        # Handle stop words and EOS token
        generation_kwargs = {
            "eos_token_id": self.pipeline.tokenizer.eos_token_id,
        }
        if stop:
            generation_kwargs["stop_sequences"] = stop

        # Handle different input types
        prompt = self.format_input(input)
        # generated_text = self.pipeline(prompt, **kwargs)[0]["generated_text"]
        outputs = self.pipeline(
            prompt,
            return_full_text=False,  # Only return new text
            **{**kwargs, **generation_kwargs}
        )
        generated_text = outputs[0]["generated_text"]

        # cleaned = re.sub(r"<(USER|SYSTEM)>.*?</\1>", "", generated_text, flags=re.DOTALL).strip()
        # if '</think>' in generated_text:
        #     response = generated_text.rsplit('</think>', 1)[-1].strip()
        # else:
        #     response = cleaned
        response = self._process_response(generated_text)
        return AIMessage(content=response)
    
    def format_input(
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
        elif isinstance(input, List):
            messages = []
            for item in input:
                # Handle objects with attributes role and content (e.g. BaseMessage)
                if hasattr(item, "role") and hasattr(item, "content"):
                    messages.append({"role": item.role, "content": item.content})
                # Handle dictionaries.
                elif isinstance(item, dict) and "role" in item and "content" in item:
                    messages.append(item)
                else:
                    return "\n".join(str(x) for x in input)
            return self.pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
    
    def _process_response(self, text: str) -> str:
        """Clean response and ensure EOS handling"""
        # Truncate at EOS token if present
        eos_token = self.pipeline.tokenizer.eos_token
        if eos_token in text:
            text = text.split(eos_token)[0]
        return text.strip()


def create_local_model(model_path, temperature=0.1):
    os.environ["HF_DATASETS_CACHE"] = model_path
    os.environ["HF_HOME"] = model_path
    os.environ["HF_HUB_CACHE"] = model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16).eval()
    tokenizer.pad_token = tokenizer.bos_token
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_new_tokens=1000,
        # do_sample=False,  # Enable greedy decoding
        # top_k=None,       # Disable top-k sampling
        top_p=0.9,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id  # Default value (ignored when do_sample=False)
    )
    return CustomHuggingFacePipeline(pipeline=hf_pipeline)


ENGINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "llama-agent": {
        "constructor": create_local_model,
        "params": {"model_path": model_path1, "temperature": 0.1},
    },
    "tablegpt-tool": {
        "constructor": create_local_model,
        "params": {"model_path": model_path2, "temperature": 0.1},
    },
    "llama-fixing": {
        "constructor": create_local_model,
        "params": {"model_path": model_path3, "temperature": 0.6},
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
