from vector_storage import VectorStorage
from transformers import BertModel, BertTokenizer
import csv
from tqdm import tqdm
import torch


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
        ).limit(500)
        embeddings = [doc["embedding"] for doc in data]
        embeddings = torch.tensor(embeddings).to(self.vector_storage.device)
        self.vector_storage.index = self.vector_storage.initialize_faiss_index(
            embeddings
        )


movie_recommender = MovieRecommender()
movie_recommender.import_data(from_file=False)
movie_recommender.build_index()
print(
    movie_recommender.search(
        "Bumbling pirate crewman  kills his captain after learning where he has hidden buried treasure. However as he begins to lose his memory, he relies more and more on the ghost of the man he's murdered to help him find the treasure. This film provided a reunion of sorts for Sellers with his onetime Goon Show colleague Spike Milligan, who appears halfway through the film."
    )
)
print("done")
