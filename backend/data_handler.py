# backend/data_handler.py
import pandas as pd
import numpy as np
import time
from fuzzywuzzy import fuzz, process

class DataHandler:
    def __init__(self, csv_path=None, df=None):
        """
        Initialize the DataHandler with either a CSV path or a DataFrame.
        Args:
            csv_path (str, optional): Path to the CSV file
            df (DataFrame, optional): Pandas DataFrame with the data
        """
        start_time = time.time()
        if df is not None:
            self.df = df
        elif csv_path:
            print("Loading CSV file...")
            self.df = pd.read_csv(csv_path)
            print(f"CSV loaded in {time.time() - start_time:.2f} seconds")
        else:
            self.df = None

        self.data_schema = {}
        if self.df is not None:
            self._analyze_schema()
        print(f"DataHandler initialization completed in {time.time() - start_time:.2f} seconds")

    def _analyze_schema(self):
        """Analyze the schema of the loaded DataFrame."""
        if self.df is None:
            return

        cols = self.df.columns.tolist()
        self.data_schema = {
            'columns': {col: str(self.df[col].dtype) for col in cols},
            'shape': self.df.shape,
            'has_country': 'Country' in cols or 'Team' in cols,
            'has_athlete': 'Athlete' in cols or 'Name' in cols,
            'has_year': any('year' in col.lower() for col in cols),
            'medal_columns': [col for col in cols if 'medal' in col.lower() or col in ['Gold', 'Silver', 'Bronze', 'Total']]
        }

    def fuzzy_search_column(self, column, search_term, threshold=70):
        """Perform fuzzy search on a column."""
        if column not in self.df.columns:
            return []

        matches = []
        column_values = self.df[column].astype(str).unique()

        if len(column_values) < 1000:
            for idx, value in enumerate(self.df[column].astype(str)):
                similarity = fuzz.token_set_ratio(value.lower(), search_term.lower())
                if similarity >= threshold:
                    matches.append(idx)
        else:
            best_matches = process.extract(search_term, column_values, scorer=fuzz.token_set_ratio, limit=10)
            good_matches = [match[0] for match in best_matches if match[1] >= threshold]
            for match_value in good_matches:
                indices = self.df[self.df[column].astype(str).str.contains(match_value, case=False, na=False)].index.tolist()
                matches.extend(indices)

        return list(set(matches))

    def search_data(self, query_params):
        """
        Search data based on query parameters from QueryProcessor.
        Args:
            query_params (dict): Dictionary of search parameters
        Returns:
            DataFrame: Filtered results
            dict: Additional information (if any)
        """
        start_time = time.time()
        if self.df is None:
            return pd.DataFrame(), {"error": "No data loaded"}

        results = self.df.copy()
        filters = query_params.get('filters', {})

        # Country filter
        if 'country' in filters and ('Country' in results.columns or 'Team' in results.columns):
            col = 'Team' if 'Team' in results.columns else 'Country'
            results = results[results[col].str.contains(filters['country'], case=False, na=False)]

        # City filter
        if 'city' in filters and 'City' in results.columns:
            results = results[results['City'].str.contains(filters['city'], case=False, na=False)]

        # Year filter
        if 'year' in filters and any('year' in col.lower() for col in results.columns):
            year_col = next(col for col in results.columns if 'year' in col.lower())
            results = results[results[year_col].astype(str).str.contains(filters['year'], na=False)]

        # Athlete filter
        if 'athlete' in filters and ('Name' in results.columns or 'Athlete' in results.columns):
            name_col = 'Name' if 'Name' in results.columns else 'Athlete'
            results = results[results[name_col].str.contains(filters['athlete'], case=False, na=False)]

        # Medal type filter
        if 'medal_type' in filters:
            medal_type = filters['medal_type']
            medal_col = None
            if 'Medal' in results.columns:
                medal_col = 'Medal'
            elif medal_type in results.columns:
                medal_col = medal_type
            elif 'Gold' in results.columns and medal_type == 'gold':
                results = results[results['Gold'] > 0]
            elif 'Silver' in results.columns and medal_type == 'silver':
                results = results[results['Silver'] > 0]
            elif 'Bronze' in results.columns and medal_type == 'bronze':
                results = results[results['Bronze'] > 0]

        # Ranking intent
        if query_params.get('intent') == 'ranking':
            medal_col = next((col for col in ['Gold', 'Silver', 'Bronze', 'Total'] if col in results.columns), None)
            if medal_col:
                ascending = query_params.get('ascending', False)
                results = results.sort_values(by=medal_col, ascending=ascending)
                limit = query_params.get('limit', 10)
                results = results.head(limit)

        # If no results found
        if results.empty:
            print(f"Data search completed in {time.time() - start_time:.2f} seconds")
            return results, {"empty": True}

        print(f"Data search completed in {time.time() - start_time:.2f} seconds")
        return results, {"record_count": len(results)}