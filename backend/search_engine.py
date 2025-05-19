import pandas as pd
from .query_processor import QueryProcessor
from .data_handler import DataHandler
import nltk

# Ensure the punkt tokenizer is available for NLTK tokenization
nltk.download('punkt')

class SearchEngine:
    def __init__(self, csv_path=None, df=None):
        """
        Initialize the search engine with data.
        
        Args:
            csv_path (str, optional): Path to the CSV file
            df (DataFrame, optional): Pandas DataFrame with the data
        """
        self.data_handler = DataHandler(csv_path=csv_path, df=df)
        self.query_processor = QueryProcessor()
        
        # If data is loaded, learn schema from it
        if csv_path or df is not None:
            self.learn_data_schema()
    
    def learn_data_schema(self):
        """Learn schema from the loaded data to help with query processing."""
        self.query_processor.learn_from_data(self.data_handler.df)
    
    def load_data(self, csv_path):
        """
        Load data from a CSV file.
        
        Args:
            csv_path (str): Path to the CSV file
        """
        df = self.data_handler.load_data(csv_path)
        self.query_processor.learn_from_data(df)
        return df
    
    def search(self, query):
        """
        Process a natural language query and search the data.
        
        Args:
            query (str): Natural language query
            
        Returns:
            tuple: (DataFrame with results, dict with additional info/analysis)
        """
        # Process the query into structured parameters
        query_params = self.query_processor.process_query(query, self.data_handler.df)
        
        # Search the data using the parameters
        results, analysis_info = self.data_handler.search_data(query_params)
        
        return results, query_params, analysis_info
    
    def format_results(self, results, query_params, analysis_info):
        """
        Format search results into a readable response.
        
        Args:
            results (DataFrame): Search results
            query_params (dict): Query parameters
            analysis_info (dict): Analysis information
            
        Returns:
            dict: Formatted results
        """
        response = {
            "original_query": query_params.get('original_query', ''),
            "result_count": len(results),
            "data": results.to_dict('records') if not results.empty else [],
            "columns": list(results.columns) if not results.empty else [],
            "intent": query_params.get('intent', 'filter'),
            "analysis": analysis_info
        }
        
        return response
    
    def process_query(self, query):
        """
        Complete end-to-end processing of a query.
        
        Args:
            query (str): Natural language query
            
        Returns:
            dict: Formatted results and analysis
        """
        results, query_params, analysis_info = self.search(query)
        formatted_response = self.format_results(results, query_params, analysis_info)
        return formatted_response