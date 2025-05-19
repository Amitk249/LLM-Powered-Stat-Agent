import pandas as pd
from backend.data_handler import DataHandler

# Sample Olympic data
data = {
    'Athlete': ['Michael Phelps', 'Usain Bolt', 'Simone Biles', 'Liu Xiang', 'Katie Ledecky'],
    'Country': ['USA', 'Jamaica', 'USA', 'China', 'USA'],
    'Year': [2008, 2012, 2016, 2008, 2020],
    'Event': ['Swimming', 'Athletics', 'Gymnastics', 'Athletics', 'Swimming'],
    'Medal': ['Gold', 'Gold', 'Gold', 'Gold', 'Gold'],
    'City': ['Beijing', 'London', 'Rio', 'Beijing', 'Tokyo']
}
df = pd.DataFrame(data)

# Initialize DataHandler
handler = DataHandler(df=df)

# Test semantic search
query = "Who won gold in 2012?"
results = handler.semantic_search(query, top_k=5)
print("Semantic Search Results:")
print(results)