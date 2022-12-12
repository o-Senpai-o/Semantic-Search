# Make sure a Milvus server is already running
from pymilvus import connections, utility
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType


def setup_DB(coll_name, dims):
    # Connect to Milvus server
    connections.connect(alias="default", host="localhost", port="19530")

    # Collection name
    collection_name = coll_name

    # Embedding size
    emb_dim = dims

    collection = create_database(collection_name, emb_dim)

    return collection


def create_database(collection_name, emb_dim):
    """
    we create collection to insert our data into database
    Once a collection has been created, we can upload our texts and vectors into it.
    
    """
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

    return collection

    
