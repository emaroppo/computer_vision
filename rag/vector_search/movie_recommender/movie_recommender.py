from rag.vector_search.movie_recommender.vector_storage import VectorStorage
from transformers import BertModel, BertTokenizer


class MovieRecommender:
    def __init__(
        self,
        db_name="movie_recommender",
        collection_name="movies",
        host="localhost",
        port=27017,
    ):
        model = BertModel.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.vector_storage = VectorStorage(
            model, tokenizer, db_name, collection_name, host, port
        )
        self.vector_storage.compute_embeddings()
        self.vector_storage.build_index()

    def search(self, query_string, k=10):
        D, I = self.vector_storage.search(query_string, k)
        return D, I

    def insert(self, data, embedding_field="text"):
        self.vector_storage.insert(data, embedding_field)

    def import_data(self, data, embedding_field="plot_summary"):
        # Import movie data into mongodb
        # read data, transform into list of dictionaries
        self.vector_storage.import_data(data, embedding_field)

        raise NotImplementedError
