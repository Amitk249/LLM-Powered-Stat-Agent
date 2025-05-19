import pytest
from backend.query_processor import QueryProcessor
import pandas as pd

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Team': ['USA', 'China', 'UK'],
        'Year': [2020, 2020, 2020],
        'Gold': [39, 38, 22],
        'Silver': [41, 32, 21],
        'Bronze': [33, 18, 22]
    })

@pytest.fixture
def query_processor(sample_data):
    processor = QueryProcessor()
    processor.learn_from_data(sample_data)
    return processor

def test_query_processor_initialization():
    processor = QueryProcessor()
    assert processor is not None
    assert processor.model is not None

def test_learn_from_data(query_processor, sample_data):
    assert len(query_processor.countries) == 3
    assert 'USA' in query_processor.countries
    assert 'China' in query_processor.countries
    assert 'UK' in query_processor.countries

def test_process_query(query_processor):
    query = "How many gold medals did USA win in 2020?"
    result = query_processor.process_query(query)
    
    assert result['intent'] in ['medal_count', 'filter']
    assert result['entities']['country'] == 'USA'
    assert result['entities']['year'] == '2020'
    assert result['entities']['medal_type'] == 'gold'

def test_semantic_matching(query_processor):
    query = "United States"
    result = query_processor._semantic_match(query, query_processor.countries)
    assert result == 'USA' 