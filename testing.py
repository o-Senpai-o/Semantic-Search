
from Dataset import process_df

papers_df = process_df("E:/Projects/nlp/Semantic search/arxiv.json")

#*************************************************************************************
print("got df")


# Make sure a Milvus server is already running
from pymilvus import connections, utility
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType

# Connect to Milvus server
connections.connect(alias="default", host="localhost", port="8200")

# Collection name
collection_name = "arxiv"

# Embedding size
emb_dim = 768


##****************************************************************************************
print("milvus connected")


# Create a schema for the collection
idx = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=16)
categories = FieldSchema(name="categories", dtype=DataType.VARCHAR, max_length=200)
title = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=4096)
abstract = FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=65535)
unix_time = FieldSchema(name="unix_time", dtype=DataType.INT64)
embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=emb_dim)

# Fields in the collection
fields = [idx, categories, title, abstract, unix_time, embedding]
schema = CollectionSchema(
    fields=fields, description="Semantic Similarity of Scientific Papers"
)

# Create a collection with the schema
collection = Collection(
    name=collection_name, schema=schema, using="default", shards_num=10
)


#*******************************************************************************************
print("schema done")



from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Scientific Papers SBERT Model
model = SentenceTransformer('allenai-specter')

def emb_gen(partition):
    return model.encode(partition['text']).tolist()


#*********************************************************************************************




# Initialize
collection = Collection(collection_name)

for partition in tqdm(range(papers_df.npartitions)):
    # Get the dask dataframe for the partition
    subset_df = papers_df.get_partition(partition)

    # Check if dataframe is empty
    if len(subset_df.index) != 0:
        # Metadata
        data = [
            subset_df[col].values.compute().tolist()
            for col in ["id", "categories", "title", "abstract", "unix_time"]
        ]

        # Embeddings
        data += [
            subset_df
            .map_partitions(emb_gen)
            .compute()[0]
        ]

        # Insert data
        collection.insert(data)


#***********************************************************************************************
print("data uploaded")

# Add an ANN index to the collection
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"efConstruction": 128, "M": 8},
}

collection.create_index(field_name="embedding", index_params=index_params)
collection.load()
#***********************************************************************************************
print("indexing the data done")


