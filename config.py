"""
Configuration settings for the Olympic Stats Dashboard application.
"""

# Data settings
DATA_PATH = "data/olympic.csv"

# Speech recognition settings
SPEECH_RECOGNITION = {
    "energy_threshold": 4000,  # Microphone sensitivity
    "pause_threshold": 0.8,    # Seconds of silence to consider end of phrase
    "timeout": 5,              # Maximum seconds to wait for audio
    "phrase_time_limit": 10    # Maximum seconds for a single phrase
}

# Vector embedding settings
EMBEDDING = {
    "model_name": "all-MiniLM-L6-v2",  # Sentence transformer model
    "cache_dir": "./.cache/embeddings"  # Cache directory for models
}

# Search settings
SEARCH = {
    "name_match_threshold": 0.75,       # Minimum similarity score (0-1) for name matches
    "query_match_threshold": 0.65,      # Minimum similarity score for query matches
    "top_k_results": 10,                # Maximum number of results to return
    "fallback_to_fuzzy": True,          # Use fuzzy matching as fallback
    "fuzzy_match_ratio": 75             # Minimum ratio (0-100) for fuzzy matching
}

# UI settings
UI = {
    "page_title": "Olympic Stats Dashboard",
    "page_icon": "🏅",
    "layout": "centered",
    "initial_sidebar_state": "expanded",
    "theme_color": "#1E90FF",           # Primary theme color
    "table_height": 400,                # Height of results table in pixels
    "max_displayed_results": 20         # Maximum number of rows to display
}

# Advanced settings
DEBUG = False                          # Enable debug output
CACHE_EMBEDDINGS = True                # Cache vector embeddings between runs