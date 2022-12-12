from search import semantic_search
from Dataset import process_df


if __name__ == "__main__":
    path = 'E:/Projects/nlp/Semantic search/arxiv.json'
    collection_name = "arxiv"
    embedding_dims = 768
    df = process_df(path)

    ss = semantic_search(path, collection_name, embedding_dims, df)

    # fist upload the data into milvus database, this will also load the collection into memory for quering
    # ss.upload_data()

    # this function will take the qeuery and dispaly the similar results
    # Query for papers that are similar to the SimCSE paper
    title = "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
    abstract = """This paper presents SimCSE, a simple contrastive learning framework that greatly advances state-of-the-art sentence embeddings. We first describe an unsupervised approach, which takes an input sentence and predicts itself in a contrastive objective, with only standard dropout used as noise. This simple method works surprisingly well, performing on par with previous supervised counterparts. We find that dropout acts as minimal data augmentation, and removing it leads to a representation collapse. Then, we propose a supervised approach, which incorporates annotated pairs from natural language inference datasets into our contrastive learning framework by using "entailment" pairs as positives and "contradiction" pairs as hard negatives. We evaluate SimCSE on standard semantic textual similarity (STS) tasks, and our unsupervised and supervised models using BERT base achieve an average of 76.3% and 81.6% Spearman's correlation respectively, a 4.2% and 2.2% improvement compared to the previous best results. We also show -- both theoretically and empirically -- that the contrastive learning objective regularizes pre-trained embeddings' anisotropic space to be more uniform, and it better aligns positive pairs when supervised signals are available."""

    query_text = f"{title}[SEP]{abstract}"
    ss.query_and_display(query_text, num_results=10)

