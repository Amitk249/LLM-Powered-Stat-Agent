# modules/response_generator.py

from llm_utils import LocalLLM
import pandas as pd
import time

class ResponseGenerator:
    def __init__(self):
        """
        Initialize the response generator with an LLM for natural language generation.
        """
        start_time = time.time()
        self.llm = LocalLLM()
        print(f"ResponseGenerator initialization completed in {time.time() - start_time:.2f} seconds")
    
    def generate_response(self, query, results, entities, intent):
        """
        Generate a natural language response based on the query and filtered results.

        Args:
            query (str): The original query from the user
            results (DataFrame): Results from DataHandler
            entities (dict): Extracted entities like country, year, medal_type
            intent (str): Intent like 'filter', 'ranking', 'analysis'

        Returns:
            str: Natural language explanation
        """
        start_time = time.time()
        if results.empty:
            response = self._generate_no_result_message(entities)
            print(f"Empty result response generated in {time.time() - start_time:.2f} seconds")
            return response

        prompt = self._build_prompt(query, results, entities, intent)
        response = self.llm.generate_response(prompt)
        print(f"Full response generation completed in {time.time() - start_time:.2f} seconds")
        return response
    
    def _build_prompt(self, query, results, entities, intent):
        """
        Build a structured prompt for the LLM based on user input and results.

        Args:
            query (str): Original user question
            results (DataFrame): Filtered dataset
            entities (dict): Matched entities (country, year, etc.)
            intent (str): Intent from QueryProcessor
            
        Returns:
            str: Prompt to send to the LLM
        """
        start_time = time.time()
        # Build context from entities
        context = []
        if entities.get('country'):
            context.append(f"Country: {entities['country']}")
        if entities.get('year'):
            context.append(f"Year: {entities['year']}")
        if entities.get('medal_type') and entities['medal_type'] != 'total':
            context.append(f"Medal Type: {entities['medal_type'].title()}")
        if entities.get('city'):
            context.append(f"Host City: {entities['city']}")
        
        # Get available columns from results
        available_columns = results.columns.tolist()
        
        # Build full prompt
        prompt = f"""
You are an Olympic assistant. Based on the following data, clearly answer the user's question.

User Question: "{query}"
Intent: {intent}
Context: {", ".join(context) if context else "No filters"}

Filtered Data:
{results.to_string(index=False) if not results.empty else "No results found"}

Answer:
"""
        print(f"Prompt building completed in {time.time() - start_time:.2f} seconds")
        return prompt

    def _generate_no_result_message(self, entities):
        """
        Return a friendly message when no data is found.

        Args:
            entities (dict): Entities extracted from query

        Returns:
            str: Friendly explanation
        """
        start_time = time.time()
        if entities.get('year'):
            response = f"🔍 I couldn't find any data for the year {entities['year']}."
        elif entities.get('country'):
            response = f"🌍 No records found for {entities['country']} in the dataset."
        elif entities.get('athlete'):
            response = f"🏃‍♂️ Sorry, I couldn't find any athlete named '{entities['athlete']}' in the dataset."
        elif entities.get('medal_type'):
            response = f"🏅 No data found related to {entities['medal_type']} medals."
        else:
            response = "🤔 I couldn't understand or find data for your request."
        
        print(f"No result message generated in {time.time() - start_time:.2f} seconds")
        return response
