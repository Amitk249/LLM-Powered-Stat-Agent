from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
from query_processor import QueryProcessor
from data_handler import DataHandler
from response_generator import ResponseGenerator
import time

app = FastAPI(title="Voice-Enabled Olympic Data Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_handler = None
query_processor = None
response_generator = None

class QueryRequest(BaseModel):
    query: str
    data_path: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    processing_time: float
    entities: Dict[str, Any]
    intent: str

@app.on_event("startup")
async def startup_event():
    global data_handler, query_processor, response_generator
    start_time = time.time()
    
    # Initialize components
    data_handler = DataHandler()
    query_processor = QueryProcessor()
    response_generator = ResponseGenerator()
    
    print(f"Application startup completed in {time.time() - start_time:.2f} seconds")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    start_time = time.time()
    
    try:
        # Load data if provided
        if request.data_path:
            df = pd.read_csv(request.data_path)
            data_handler.df = df
            query_processor.learn_from_data(df)
        
        # Process query
        query_params = query_processor.process_query(request.query)
        
        # Get results
        results, info = data_handler.search_data(query_params)
        
        # Generate response
        response = response_generator.generate_response(
            request.query,
            results,
            query_params['entities'],
            query_params['intent']
        )
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            response=response,
            processing_time=processing_time,
            entities=query_params['entities'],
            intent=query_params['intent']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 