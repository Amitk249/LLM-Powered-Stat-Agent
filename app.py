# app.py
import streamlit as st
import pandas as pd
import os
import sys

# Add the 'modules' folder to Python path
MODULES_PATH = os.path.join(os.path.dirname(__file__), "modules")
if MODULES_PATH not in sys.path:
    sys.path.append(MODULES_PATH)

# Now these imports should work
from backend.query_processor import QueryProcessor
from backend.data_handler import DataHandler
from backend.voice_input import get_voice_input
from backend.response_generator import ResponseGenerator

# Streamlit UI Setup
st.title("🏅 Olympic Voice Assistant")
st.subheader("Ask questions about athletes, countries, and events")

# Upload CSV
uploaded_file = st.file_uploader("Upload your Olympic dataset", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded")
    
    # Add debugging information
    st.write("### Dataset Information")
    st.write("Columns:", df.columns.tolist())
    st.write("Shape:", df.shape)
    st.write("Sample Data:")
    st.dataframe(df.head())

    # Initialize processors
    processor = QueryProcessor()
    handler = DataHandler(df=df)
    responder = ResponseGenerator()

    # Learn from data
    processor.learn_from_data(df)

    if st.button("🎙️ Speak"):
        with st.spinner("Listening..."):
            user_query = get_voice_input()
            if user_query:
                st.write(f"🗣️ You said: `{user_query}`")
                query_params = processor.process_query(user_query, df=df)
                results, analysis_info = handler.search_data(query_params)

                st.markdown("### 🔍 Results")
                if not results.empty:
                    st.dataframe(results.head())
                else:
                    st.info("No results found.")

                st.markdown("### 🤖 Reasoning")
                response = responder.generate_response(
                    query=user_query,
                    results=results,
                    entities=query_params['entities'],
                    intent=query_params['intent']
                )
                st.markdown(response)
else:
    st.info("📂 Please upload a CSV file")