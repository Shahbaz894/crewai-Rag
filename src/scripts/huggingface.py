
# scripts/huggingface.py
import os
import requests
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any

load_dotenv()

logger = logging.getLogger(__name__)
# ... (your logger configuration)

class HuggingFaceLLM:
    def __init__(self):
        try:
            self.api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not self.api_key:
                logger.error("HUGGINGFACE_API_KEY is missing. Set it in the environment variables.")
                raise ValueError("HUGGINGFACE_API_KEY is missing.")

            self.model = "google/gemma-7b"  # Or your preferred default model
            self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"

            logger.info(f"HuggingFaceLLM initialized with model: {self.model}")
        except Exception as e:
            logger.exception(f"Initialization failed: {str(e)}")
            raise

    def generate_response(self, context: str, query: str) -> Optional[str]:
        try:
            prompt = self._create_prompt(context, query)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 700,  # Adjust as needed
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True
                }
            }

            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)

            if response.status_code != 200:
                logger.error(f"API request failed: {response.status_code}, {response.text}")
                return None

            return self._parse_response(response.json())

        except requests.exceptions.Timeout:
            logger.error("Request timed out.")
            return None
        except Exception as e:
            logger.exception(f"Error in generating response: {str(e)}")
            return None

    def _create_prompt(self, context: str, query: str) -> str:
        return f"""Please provide a comprehensive and accurate answer to the medical question based on the following context and query.
        If the context doesn't contain relevant information, please indicate that clearly.

        Context: {context}

        Question: {query}

        Answer:"""

    def _parse_response(self, response: Any) -> Optional[str]:
        try:
            if isinstance(response, list) and response:
                return response[0].get('generated_text', '').strip()
            elif isinstance(response, dict):
                return response.get('generated_text', '').strip()
            elif isinstance(response, str):  # Handle string responses
                return response.strip()
            return None  # Return None if the response structure is not as expected
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return None

    def set_model(self, model_name: str) -> None:
        try:
            self.model = model_name
            self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            logger.info(f"Model changed to: {model_name}")
        except Exception as e:
            logger.exception(f"Error changing model: {str(e)}")
            raise