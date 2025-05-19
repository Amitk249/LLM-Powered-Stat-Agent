'''
Connecting with query processor

'''

# voice_input.py
import time
import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from query_processor import QueryProcessor
from data_handler import DataHandler
import pandas as pd

# Load Whisper model
model = whisper.load_model("base")

# Sample data for testing
df = pd.read_csv("/media/amit/009FA5F2572F49B8/voice_embedding/data/olympic.csv")
processor = QueryProcessor()
handler = DataHandler(df=df)

def record_audio(duration=5, fs=44100):
    start_time = time.time()
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    recording_time = time.time() - start_time
    print(f"Recording finished. Time taken: {recording_time:.2f} seconds")
    return audio

def save_audio_to_wav(audio, filename="output.wav", fs=44100):
    write(filename, fs, audio)

def transcribe_audio_with_whisper(filename="output.wav"):
    start_time = time.time()
    result = model.transcribe(filename)
    transcription_time = time.time() - start_time
    print(f"Transcription time: {transcription_time:.2f} seconds")
    return result["text"]

def get_voice_input():
    total_start_time = time.time()
    print("listening")
    audio = record_audio()
    save_audio_to_wav(audio)
    text = transcribe_audio_with_whisper()
    total_time = time.time() - total_start_time
    print(f"You said: {text}")
    print(f"Total processing time: {total_time:.2f} seconds")
    return text

if __name__ == "__main__":
    from response_generator import ResponseGenerator
    responder = ResponseGenerator()

    while True:
        user_input = input("Type 'speak' to speak or 'exit': ").lower()
        if user_input == "exit":
            break
        elif user_input == "speak":
            query_start_time = time.time()
            query = get_voice_input()
            if query:
                # Process the query
                query_params = processor.process_query(query, df=df)
                results, analysis_info = handler.search_data(query_params)

                # Generate natural language response
                response = responder.generate_response(
                    query=query,
                    results=results,
                    entities=query_params['entities'],
                    intent=query_params['intent']
                )

                total_query_time = time.time() - query_start_time
                print("Assistant:", response)
                print(f"Total query processing time: {total_query_time:.2f} seconds")