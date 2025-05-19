# from modules.query_processor import QueryProcessor
# import pandas as pd

# # Initialize the processor
# processor = QueryProcessor()

# # Example data (you can load your actual dataset here)
# sample_data = pd.DataFrame({
#     'Country': ['USA', 'China'],
#     'Year': [2020, 2020],
#     'Gold': [5, 3]
# })

# # Learn from the data (optional)
# processor.learn_from_data(sample_data)

# # Process a query
# query = "Who won gold in 2020?"
# result = processor.process_query(query, df=sample_data)

# print(result)


from backend.query_processor import QueryProcessor
# from backend.data_handler import DataHandler
# import pandas as pd

# Load the sample dataset
import pandas as pd
from backend.data_handler import DataHandler

# Use only the small test DataFrame
small_df = pd.DataFrame({
    'Country': ['USA', 'China'],
    'Year': [2008, 2012],
    'Medal': ['Gold', 'Gold']
})

handler = DataHandler(df=small_df)
processor = QueryProcessor()
processor.learn_from_data(small_df)

query = "Who won gold in 2008?"
params = processor.process_query(query, small_df)
results, info = handler.search_data(params)

print(results)

# # Test Query 2: "Show me top 5 athletes from USA by total medals"
# query2 = "Show me top 5 athletes from USA by total medals"
# result2 = processor.process_query(query2, df=df)
# print("\nQuery 2 Result:")
# print(result2)

# # Test Query 3: "Which city hosted the most gold medals in 2008?"
# query3 = "Which city hosted the most gold medals in 2008?"
# result3 = processor.process_query(query3, df=df)
# print("\nQuery 3 Result:")
# print(result3)

# # Test Query 4: "Analyze China's performance in Athletics"
# query4 = "Analyze China's performance in Athletics"
# result4 = processor.process_query(query4, df=df)
# print("\nQuery 4 Result:")
# print(result4)