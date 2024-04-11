from pymongo import MongoClient
import numpy as np
import faiss
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class VectorStorage:
    def __init__(
        self,
        model_name,
        db_name,
        collection_name,
        host="localhost",
        port=27017,
    ):
        self.client = MongoClient(host, port)
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection = self.client[self.db_name][self.collection_name]

        # Use GPU if it's available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name).to(self.device)

    def load_embeddings(self):
        embeddings = []
        for doc in self.collection.find():
            embeddings.append(doc["embedding"])
        return np.array(embeddings)

    def encode(self, text):
        with torch.no_grad():
            embeddings = self.model.encode(text, convert_to_tensor=True)
        return embeddings

    def initialize_faiss_index(self, embeddings):
        dimension = embeddings.shape[1]  # Assuming embeddings are 1D arrays
        lengths = [len(embedding) for embedding in embeddings]
        print(set(lengths))
        print(f"Embedding dimension: {dimension}")
        index = faiss.IndexFlatL2(dimension)
        print("Index type: ", type(index))
        index.add(embeddings)
        return index

    def search(self, query_string, k=10):
        query_embedding = self.model.encode(query_string)
        D, I = self.index.search(query_embedding.reshape(1, -1), k=k)
        return D, I

    def compute_embeddings(self, embedding_field="plot_summary"):
        for doc in self.collection.find({"embedding": {"$exists": False}}):
            embedding = self.encode(doc[embedding_field]).to(self.device).tolist()
            self.collection.update_one(
                {"_id": doc["_id"]}, {"$set": {"embedding": embedding}}
            )
        embeddings = self.load_embeddings()

    def insert(self, data, embedding_field="plot_summary"):
        if "embedding" not in data:
            data["embedding"] = (
                self.encode(data[embedding_field]).to(self.device).tolist()
            )
        self.collection.insert_one(data)

    def import_data(self, data, embedding_field="plot_summary"):
        for doc in tqdm(data):
            if "embedding" not in doc:
                doc["embedding"] = (
                    self.encode(doc[embedding_field]).to(self.device).tolist()
                )
                self.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"embedding": doc["embedding"]}},
                    upsert=True,
                )
