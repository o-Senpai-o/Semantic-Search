
import dask.bag as db
import json
from datetime import datetime
import time

from preprocess import filters, text_col, v1_date



# Make sure a Milvus server is already running
from pymilvus import connections, utility
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType


#----------------------------------------------------------------------------


def process_df(file_path):

    data_path = file_path

    # Read the file in blocks of 20MB and parse the JSON.
    papers_db = db.read_text(data_path, blocksize="10MB").map(json.loads)

    # Print the first row
    papers_db.take(1)


    # Specify columns to keep in the final table
    cols_to_keep = ["id", "categories", "title", "abstract", "unix_time", "text"]

    # Apply the pre-processing, Dash supports chaining functions so we will use 
    # map adn filters functon to preprocess the data
    papers_db = (
        papers_db.map(lambda row: v1_date(row))
        .map(lambda row: text_col(row))
        .map(
            lambda row: {
                key: value 
                for key, value in row.items() 
                if key in cols_to_keep
            }
        )
        .filter(filters)
    )

    # Print the first row
    papers_db.take(1)

    # Convert the Dask Bag to a Dask Dataframe, we can use this dataframe using pandas also
    # which will be easy for us to preprocess further if required
    schema = {
        "id": str,
        "title": str,
        "categories": str,
        "abstract": str,
        "unix_time": int,
        "text": str,
    }
    papers_df = papers_db.to_dataframe(meta=schema)

    # Display first 5 rows
    print(papers_df.head())

    return papers_df




# if __name__ == "__main__":
#     papers_df = process_df("E:/Projects/nlp/Semantic search/arxiv.json")
#     print(papers_df.columns)


#     # Connect to Milvus server
#     connections.connect(alias="default", host="localhost", port="19530")

#     # Collection name
#     collection_name = "arxiv"

#     # Embedding size
#     emb_dim = 768


#     ##****************************************************************************************
#     print("milvus connected")


#     # Create a schema for the collection
#     idx = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=16)
#     categories = FieldSchema(name="categories", dtype=DataType.VARCHAR, max_length=200)
#     title = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=4096)
#     abstract = FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=65535)
#     unix_time = FieldSchema(name="unix_time", dtype=DataType.INT64)
#     embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=emb_dim)

#     # Fields in the collection
#     fields = [idx, categories, title, abstract, unix_time, embedding]
#     schema = CollectionSchema(
#         fields=fields, description="Semantic Similarity of Scientific Papers"
#     )

#     # Create a collection with the schema
#     collection = Collection(
#         name=collection_name, schema=schema, using="default", shards_num=10
#     )


#     #*******************************************************************************************
#     print("schema done")



#     from sentence_transformers import SentenceTransformer
#     from tqdm import tqdm

#     # Scientific Papers SBERT Model
#     model = SentenceTransformer('allenai-specter')

#     def emb_gen(partition):
#         return model.encode(partition['text']).tolist()


#     #*********************************************************************************************




#     # Initialize
#     collection = Collection(collection_name)
#     count = 0
#     for partition in tqdm(range(papers_df.npartitions)):
#         # Get the dask dataframe for the partition
#         subset_df = papers_df.get_partition(partition)

#         # Check if dataframe is empty
#         if count > 50:
#             break
#         if len(subset_df.index) != 0:
#             # Metadata
#             data = [
#                 subset_df[col].values.compute().tolist()
#                 for col in ["id", "categories", "title", "abstract", "unix_time"]
#             ]

#             # Embeddings
#             data += [
#                 subset_df
#                 .map_partitions(emb_gen)
#                 .compute()[0]
#             ]

#             # Insert data
#             collection.insert(data)
#             print("inserted partition 1")
#         count += 1


#     #***********************************************************************************************
#     print("data uploaded")

#     # Add an ANN index to the collection
#     index_params = {
#         "metric_type": "L2",
#         "index_type": "HNSW",
#         "params": {"efConstruction": 128, "M": 8},
#     }

#     collection.create_index(field_name="embedding", index_params=index_params)
#     collection.load()
#     #***********************************************************************************************
#     print("indexing the data done")

