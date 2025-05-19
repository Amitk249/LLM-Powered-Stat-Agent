# test_llm.py
from backend.llm_utils import LocalLLM

# Initialize the LLM
llm = LocalLLM()

# Try a simple prompt
prompt = "Explain how Olympic medals are awarded."
response = llm.generate_response(prompt)

print("Prompt:", prompt)
print("Response:", response)