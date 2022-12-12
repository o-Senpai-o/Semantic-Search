from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import datetime
from datetime import datetime

from pymilvus import connections, utility
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType

from Dataset import process_df
from vector_database import setup_DB



# path = 'E:/Projects/nlp/Semantic search/arxiv.json'



class semantic_search:

    def __init__(self, path, collection_name, embedding_dims, df):
        self.path = path

        # lets first process the dataset, which is in the json format
        self.papers_df  = df

        # after we get our processed partitioned dataframe, we setup or database to  store the data
        self.collection_name = collection_name           #"arxiv"
        self.embedding_dim = embedding_dims              # 768


        # Scientific Papers SBERT Model
        # we use SPECTER which is a model trained on scientific reasearch ppapers
        # SPECTER : Scientific Paper Embeddings using Citation-informed TransformERs is a model 
        #           to convert scientific papers to embeddings.
        # link : https://arxiv.org/abs/2004.07180
        self.model = SentenceTransformer('allenai-specter')

        # the main collection in milvus
        self.collection = setup_DB(self.collection_name, self.embedding_dim)

    
    def emb_gen(self, partition):
        """
        get dask dataframe partitions 
        and create embedding for text columns
        """
        return self.model.encode(partition['text']).tolist()


    def upload_data(self):
        count = 0
        for partition in tqdm(range(self.papers_df.npartitions)):
            if count > 30:
                continue
        
            # Get the dask dataframe for the partition
            subset_df = self.papers_df.get_partition(partition)

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
                    .map_partitions(self.emb_gen)
                    .compute()[0]
                ]

                # Insert data
                self.collection.insert(data)
        
            count += 1
        
            
        
        # Add an ANN index to the collection
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"efConstruction": 128, "M": 8},
        }

        self.collection.create_index(field_name="embedding", index_params=index_params)

        # Finally, the data in our Milvus collection is ready to be queried. 
        # First, we must load the collection into memory to run queries against it.

        self.collection.load()
    

    def query_and_display(self, query_text, num_results=10):
        # Embed the Query Text
        query_emb = [self.model.encode(query_text)]

        # Search Params
        search_params = {"metric_type": "L2", "params": {"ef": 128}}

        # Search
        query_start = datetime.now()
        results = self.collection.search(
            data=query_emb,
            anns_field="embedding",
            param=search_params,
            limit=num_results,
            expr=None,
            output_fields=["title", "abstract"],
        )
        query_end = datetime.now()

        # Print Results
        print(f"Query Speed: {(query_end - query_start).total_seconds():.2f} s")
        print("Results:")
        for res in results[0]:
            title = res.entity.get("title").replace("\n ", "")
            print(f"➡️ ID: {res.id}. L2 Distance: {res.distance:.2f}")
            print(f"Title: {title}")
            print(f"Abstract: {res.entity.get('abstract')}")


