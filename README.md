<<<<<<< HEAD
# LLM-Powered-Stat-Agent
=======
# Voice-Enabled Olympic Data Assistant

A voice-enabled assistant that helps users query Olympic data using natural language processing and local LLM capabilities.

## Features

- Voice input processing
- Natural language query understanding
- Semantic search capabilities
- Local LLM integration (Mistral-7B)
- Olympic data analysis and visualization

## Project Structure

```
voice_embedding/
├── backend/
│   ├── data_handler.py
│   ├── query_processor.py
│   ├── response_generator.py
│   └── llm_utils.py
├── frontend/
│   └── voice_input.py
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/voice_embedding.git
cd voice_embedding
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

2. Run the voice input script:
```bash
python frontend/voice_input.py
```

## Development

- Code formatting: `black .`
- Linting: `flake8`
- Import sorting: `isort .`
- Running tests: `pytest`

## Performance Monitoring

The application includes built-in timing functionality to monitor:
- Model loading time
- Query processing time
- Semantic search performance
- Response generation time

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 
>>>>>>> 56e6bb8 (Initial commit: Voice-Enabled Data Assistant)
