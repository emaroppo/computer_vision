from vector_storage import VectorStorage
from sentence_transformers import SentenceTransformer
import csv
from tqdm import tqdm
import numpy as np
import torch


class MovieRecommender:
    def __init__(
        self,
        db_name="movie_recommender",
        model_name="all-MiniLM-L6-v2",
        collection_name="movies",
        host="localhost",
        port=27017,
    ):

        self.vector_storage = VectorStorage(
            model_name, db_name, collection_name, host, port
        )
        self.build_index()

    def search(self, query_string, k=10):
        D, I = self.vector_storage.search(query_string, k)
        return D, I

    def insert(self, data, embedding_field="plot_summary"):
        self.vector_storage.insert(data, embedding_field)

    def import_data(self, embedding_field="plot_summary", from_file=False):

        if from_file:
            print("Importing data from file")
            with open(
                "rag/vector_search/movie_recommender/MovieSummaries/plot_summaries.txt",
                "r",
            ) as file:
                reader = csv.reader(file, delimiter="\t")
                for row in tqdm(reader):
                    self.vector_storage.collection.insert_one(
                        {"movie_id": row[0], "plot_summary": row[1]}
                    )

            # add movie metadata to the database, matching the movie summaries to the movie metadata using the movie ID
            with open(
                "rag/vector_search/movie_recommender/MovieSummaries/movie.metadata.tsv",
                "r",
            ) as file:
                reader = csv.reader(file, delimiter="\t")
                for row in tqdm(reader):
                    entry = dict()
                    entry["movie_id"] = row[0]
                    entry["freebase_id"] = row[1]
                    entry["title"] = row[2]
                    entry["release_date"] = row[3]
                    if row[4] != "":
                        entry["revenue"] = float(row[4])
                    if row[5] != "":
                        entry["runtime"] = float(row[5])
                    entry["languages"] = row[6]
                    entry["countries"] = row[7]
                    entry["genres"] = row[8]
                    self.vector_storage.collection.update_one(
                        {"movie_id": entry["movie_id"]}, {"$set": entry}, upsert=True
                    )
        data = list(self.vector_storage.collection.find().limit(500))
        print(len(data))
        self.vector_storage.import_data(data, embedding_field)

    def build_index(self):
        data = self.vector_storage.collection.find(
            {"embedding": {"$exists": True}}
        ).sort("title")
        data = list(data)
        embeddings = [doc["embedding"] for doc in data]
        embeddings = np.array(embeddings).astype(np.float32)

        self.vector_storage.index = self.vector_storage.initialize_faiss_index(
            embeddings
        )

    def semantic_search(self, query_string, k=10):
        D, I = self.search(query_string, k)
        movies = self.vector_storage.collection.find(
            {"embedding": {"$exists": True}}
        ).sort("title")
        movies = list(movies)
        movies = [movies[i] for i in I[0]]

        return D, I, movies

    def similarity_search(self, movie_title, k=10):
        movie = self.vector_storage.collection.find_one({"title": movie_title})
        if movie is None:
            return None
        query_embedding = np.array(movie["embedding"]).astype(np.float32)
        D, I = self.vector_storage.index.search(query_embedding.reshape(1, -1), k=k)
        movies = self.vector_storage.collection.find(
            {"embedding": {"$exists": True}}
        ).sort("title")
        movies = list(movies)
        movies = [movies[i] for i in I[0]]

        return D, I, movies
