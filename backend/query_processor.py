import re
import nltk
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz, process
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

class QueryProcessor:
    def __init__(self, data_schema=None):
        """
        Initialize the QueryProcessor with dataset schema information.
        
        Args:
            data_schema (dict, optional): Dictionary containing column names and types
        """
        self.data_schema = data_schema
        self.stopwords = set(stopwords.words('english'))
        
        # Time model loading
        start_time = time.time()
        print("Loading the model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
        end_time = time.time()
        print(f"Model loaded in {end_time - start_time:.2f} seconds")

        # Define query pattern templates for intent classification
        self.intent_templates = {
            'ranking': "Show top countries or athletes",
            'medal_count': "How many medals did someone win",
            'filter': "Tell me about an athlete or event",
            'analysis': "Analyze performance or compare stats"
        }

        self.intent_embeddings = {
            k: self.model.encode([v], convert_to_tensor=True)
            for k, v in self.intent_templates.items()
        }

        # These will be populated based on data
        self.countries = []
        self.cities = []
        self.years = []
        self.all_entities = {}
        self.kb_embeddings = {}

    def learn_from_data(self, df):
        """
        Learn possible entities (countries, cities, years) from the dataset.

        Args:
            df (DataFrame): The dataset to learn from
        """
        if df is None:
            return

        # Store dataset for reference
        self.df = df

        # Extract unique values for key columns
        if 'Team' in df.columns:
            self.countries = df['Team'].dropna().unique().tolist()
        elif 'Country' in df.columns:
            self.countries = df['Country'].dropna().unique().tolist()
        self.all_entities['country'] = self.countries
        self.kb_embeddings['country'] = self.model.encode(self.countries, convert_to_tensor=True) if self.countries else []

        if 'City' in df.columns:
            self.cities = df['City'].dropna().unique().tolist()
            self.all_entities['city'] = self.cities
            self.kb_embeddings['city'] = self.model.encode(self.cities, convert_to_tensor=True) if self.cities else []

        year_cols = [col for col in df.columns if 'year' in col.lower()]
        if year_cols:
            self.years = df[year_cols[0]].dropna().astype(int).unique().tolist()
            self.all_entities['year'] = [str(y) for y in self.years]

        name_col = 'Name' if 'Name' in df.columns else 'Athlete'
        if name_col in df.columns:
            self.athletes = df[name_col].dropna().unique().tolist()
            self.all_entities['athlete'] = self.athletes
            self.kb_embeddings['athlete'] = self.model.encode(self.athletes, convert_to_tensor=True) if self.athletes else []

        # Medal type knowledge base
        self.all_entities['medal_type'] = ['gold', 'silver', 'bronze', 'total']

    def preprocess_query(self, query):
        """
        Preprocess the query by removing stopwords and normalizing text.

        Args:
            query (str): The query to process

        Returns:
            str: Preprocessed query
        """
        query = query.lower()
        tokens = word_tokenize(query)
        important_stopwords = {'in', 'by', 'with', 'most', 'least', 'how', 'many', 'which', 'what', 'who'}
        filtered_tokens = [token for token in tokens if token not in self.stopwords or token in important_stopwords]
        return ' '.join(filtered_tokens)

    def _semantic_match(self, query, candidates, threshold=0.6):
        """
        Match a query to the best candidate using semantic similarity.

        Args:
            query (str): The user's query
            candidates (list): List of possible options (e.g., countries, cities)
            threshold (float): Minimum similarity score to consider a match

        Returns:
            str or None: Best-matched entity or None
        """
        if not candidates:
            return None

        start_time = time.time()
        q_emb = self.model.encode([query], convert_to_tensor=False)
        c_emb = self.model.encode(candidates, convert_to_tensor=False)
        scores = np.dot(q_emb, c_emb.T).flatten()
        idx = np.argmax(scores)
        best_score = scores[idx]
        end_time = time.time()
        
        print(f"Semantic matching completed in {end_time - start_time:.2f} seconds")
        return candidates[idx] if best_score > threshold else None

    def match_entities(self, query):
        """
        Match entities like country, city, year, athlete, etc. using semantic search.

        Args:
            query (str): The preprocessed query

        Returns:
            dict: Matched entities by type
        """
        entities = {
            'country': None,
            'year': None,
            'medal_type': None,
            'comparison': None,
            'quantity': None,
            'analysis': False,
            'city': None,
            'athlete': None
        }

        # Use semantic matching for known categories
        if 'country' in self.all_entities:
            entities['country'] = self._semantic_match(query, self.all_entities['country'])

        if 'city' in self.all_entities:
            entities['city'] = self._semantic_match(query, self.all_entities['city'])

        if 'athlete' in self.all_entities:
            entities['athlete'] = self._semantic_match(query, self.all_entities['athlete'])

        # Year extraction
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        if year_match:
            entities['year'] = year_match.group()

        # Quantity extraction
        quantity_match = re.search(r'\b\d+\b', query)
        if quantity_match:
            num = int(quantity_match.group())
            if 1800 < num < 2050:
                entities['year'] = str(num)
            else:
                entities['quantity'] = num

        # Medal type extraction
        for medal in ['gold', 'silver', 'bronze', 'total']:
            if medal in query.lower():
                entities['medal_type'] = medal
                break

        # Analysis request detection
        if any(term in query.lower() for term in ['analyze', 'summary', 'statistics']):
            entities['analysis'] = True

        return entities

    def determine_query_intent(self, query, entities):
        """
        Classify intent using semantic similarity.

        Args:
            query (str): The original query
            entities (dict): Extracted entities

        Returns:
            str: Intent of the query
        """
        q_emb = self.model.encode([query], convert_to_tensor=True)
        max_sim = -1
        best_intent = 'filter'

        for intent, emb in self.intent_embeddings.items():
            sim = float(np.dot(q_emb[0], emb[0]) / (np.linalg.norm(q_emb[0]) * np.linalg.norm(emb[0])))
            if sim > max_sim:
                max_sim = sim
                best_intent = intent

        return best_intent if max_sim > 0.6 else 'filter'

    def process_query(self, query, df=None):
        """
        Process a natural language query into structured parameters.

        Args:
            query (str): The natural language query
            df (DataFrame, optional): Dataset to learn from if not already learned

        Returns:
            dict: Parameters for searching the data
        """
        start_time = time.time()
        
        if df is not None and (self.data_schema is None or not self.all_entities.get('country')):
            self.learn_from_data(df)

        if df is not None:
            self.df = df

        preprocessed_query = self.preprocess_query(query)
        entities = self.match_entities(preprocessed_query)
        intent = self.determine_query_intent(query, entities)

        query_params = {
            'intent': intent,
            'filters': {},
            'entities': entities,
            'original_query': query
        }

        if entities['country']:
            query_params['filters']['country'] = entities['country']
        if entities['year']:
            query_params['filters']['year'] = entities['year']
        if entities['medal_type']:
            query_params['filters']['medal_type'] = entities['medal_type']
        if entities['city']:
            query_params['filters']['city'] = entities['city']
        if entities['athlete']:
            query_params['filters']['athlete'] = entities['athlete']
        if entities['quantity']:
            query_params['filters']['limit'] = entities['quantity']

        end_time = time.time()
        print(f"Total query processing time: {end_time - start_time:.2f} seconds")
        
        return query_params