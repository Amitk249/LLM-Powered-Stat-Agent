# modules/llm_utils.py

from ctransformers import AutoModelForCausalLM
import os
import time
from huggingface_hub import hf_hub_download

class LocalLLM:
    def __init__(self, model_folder="models", model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        """
        Initialize a local LLM (e.g., Mistral) using ctransformers and GGUF format.
        
        Args:
            model_folder (str): Folder containing the GGUF model
            model_file (str): Name of the GGUF file
        """
        start_time = time.time()
        # Build the full path to the model file
        os.makedirs(model_folder, exist_ok=True)
        self.model_path = os.path.join(model_folder, model_file)

        # Download the model if it doesn't exist locally
        if not os.path.isfile(self.model_path):
            print(f"Downloading model to {self.model_path}...")
            download_start = time.time()
            try:
                hf_hub_download(
                    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                    filename=model_file,
                    local_dir=model_folder,
                    local_dir_use_symlinks=False
                )
                print(f"Model downloaded in {time.time() - download_start:.2f} seconds")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {str(e)}")

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # Load the local model
        print("Loading the model...")
        load_start = time.time()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=self.model_path,
            model_type="mistral",
            context_length=2048,
            max_new_tokens=256,
            temperature=0.7
        )
        print(f"Model loaded in {time.time() - load_start:.2f} seconds")
        print(f"Total initialization time: {time.time() - start_time:.2f} seconds")

    def generate_response(self, prompt):
        """
        Generate a natural language response from the LLM.
        
        Args:
            prompt (str): Prompt to send to the LLM
            
        Returns:
            str: Generated response
        """
        try:
            start_time = time.time()
            response = self.llm(prompt)
            print(f"Response generation time: {time.time() - start_time:.2f} seconds")
            return response.strip()
        except Exception as e:
            return f"[Error] Failed to generate response: {str(e)}"