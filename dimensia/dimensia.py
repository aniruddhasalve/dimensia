import os
import pickle
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dimensia.utils import ensure_directory_exists
from dimensia.exceptions import (
    CollectionAlreadyExistsError, CollectionNotFoundError, DocumentAlreadyExistsError,
    DatabaseLoadError, DatabaseSaveError, EmbeddingModelNotSetError,
    DocumentNotFoundError
)
import ssl
import requests

class Dimensia:
    """
    Dimensia - A Vector-Based Database

    Dimensia is a vector-based database for handling documents with embeddings.
    It supports creating collections, adding documents with optional metadata, searching for documents,
    and retrieving document information.

    """
    def __init__(self, db_path=None):
        
        """
        Initializes the Dimensia database.

        Args:
            db_path (str, optional): Path to the database directory. Defaults to "dimensia_data".
        """
         
        self.db_path = db_path or "dimensia_data"
        self.collections = {}
        self.embedding_model = None
        self.set_embedding_model()
        self._load_db()

    def set_embedding_model(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Sets the embedding model used for generating document embeddings.

        Args:
            model_name (str): Name of the embedding model from `sentence-transformers`.
                              Defaults to "sentence-transformers/all-MiniLM-L6-v2".

        Raises:
            EmbeddingModelNotSetError: If the model cannot be initialized.
        """

        try:
            requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
            ssl._create_default_https_context = ssl._create_unverified_context

            self.embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbeddingModelNotSetError(f"Error setting embedding model: {str(e)}")

    def _load_db(self):
        
        """
        Loads the database from the file system. If the database does not exist,
        a new database is created.

        Raises:
            DatabaseLoadError: If there is an error loading the database.
        """

        ensure_directory_exists(self.db_path)
        db_file = os.path.join(self.db_path, "data.dim")


        if os.path.exists(db_file):
            try:
                with open(db_file, "rb") as f:
                    self.collections = pickle.load(f)
            except Exception as e:
                raise DatabaseLoadError(f"Error loading database: {str(e)}")
        else:
            self._save_db()

    def _save_db(self):
        """
        Saves the current state of the database to the file system.

        Raises:
            DatabaseSaveError: If there is an error saving the database.
        """

        db_file = os.path.join(self.db_path, "data.dim")
        try:
            with open(db_file, "wb") as f:
                pickle.dump(self.collections, f)
        except Exception as e:
            raise DatabaseSaveError(f"Error saving database: {str(e)}")

    def create_collection(self, collection_name):
        """
        Creates a new collection in the database.

        Args:
            collection_name (str): Name of the collection to create.

        Raises:
            CollectionAlreadyExistsError: If the collection already exists.
        """

        if collection_name in self.collections:
            raise CollectionAlreadyExistsError(f"Collection '{collection_name}' already exists.")
        self.collections[collection_name] = {
            "documents": [],
            "next_id": 1,
            "texts": set(),
            "metadata": {}
        }
        self._save_db()

    def add_documents(self, collection_name, documents, metadata=None):
        """
        Adds multiple documents to a specified collection.

        Args:
            collection_name (str): Name of the collection.
            documents (list): List of documents, where each document is a dictionary
                              containing at least a 'content' key.
            metadata (dict, optional): Metadata for the documents. Defaults to None.

        Raises:
            CollectionNotFoundError: If the specified collection does not exist.
        """

        if collection_name not in self.collections:
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")

        collection = self.collections[collection_name]
        metadata = metadata or {}

        existing_hashes = set(self._get_text_hash(doc['content']) for doc in collection['documents'])

        with ThreadPoolExecutor() as executor:
            futures = []
            for doc in documents:
                doc_content_hash = self._get_text_hash(doc['content'])

                if doc_content_hash in existing_hashes:
                    print(f"Duplicate document found with content: {doc['content']}.")
                    continue

                if 'id' not in doc:
                    doc['id'] = collection['next_id']
                    collection['next_id'] += 1

                futures.append(executor.submit(self._process_document, doc, collection, metadata))

            # Wait for all documents to be processed
            for future in tqdm(futures, desc="Ingesting documents", unit="document"):
                future.result()

        self._save_db()

    def _process_document(self, doc, collection, metadata):
        """
        Processes a single document by generating its embedding and adding it to the collection.

        Args:
            doc (dict): The document to process.
            collection (dict): The collection where the document will be stored.
            metadata (dict): Metadata associated with the document.

        Raises:
            EmbeddingModelNotSetError: If the embedding model is not set.
            DocumentAlreadyExistsError: If a document with the same ID already exists.
        """

        if self.embedding_model is None:
            raise EmbeddingModelNotSetError("Embedding model is not set.")

        if 'id' not in doc:
            doc['id'] = collection['next_id']
            collection['next_id'] += 1

        if any(existing_doc['id'] == doc['id'] for existing_doc in collection['documents']):
            raise DocumentAlreadyExistsError(f"Document with ID '{doc['id']}' already exists.")

        vector = self._generate_embedding(doc['content'])
        doc['vector'] = vector
        doc['metadata'] = metadata.get(doc['id'], {})
        collection['documents'].append(doc)
        collection['texts'].add(self._get_text_hash(doc['content']))

    def _get_text_hash(self, text):
        """
        Generates a hash for the given text to identify duplicates.

        Args:
            text (str): The document text.

        Returns:
            str: The SHA-256 hash of the text.
        """

        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _generate_embedding(self, text):
        """
        Generates an embedding for a given text.

        Args:
            text (str): The input text.

        Returns:
            list: A vector representing the embedding of the text.

        Raises:
            EmbeddingModelNotSetError: If the embedding model is not set.
        """

        if self.embedding_model is None:
            raise EmbeddingModelNotSetError("Embedding model is not set.")
        vector = self.embedding_model.encode(text).tolist()
        return vector

    def get_structure(self, collection_name):
        """
        Returns the structure of a collection.

        Args:
            collection_name (str): Name of the collection.

        Returns:
            dict: A dictionary containing collection structure details.

        Raises:
            CollectionNotFoundError: If the specified collection does not exist.
        """

        if collection_name not in self.collections:
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")
        collection = self.collections[collection_name]
        return {
            "document_count": len(collection["documents"]),
            "vector_size": len(collection["documents"][0]['vector']) if collection["documents"] else 0,
            "metadata": collection["metadata"]
        }

    def get_vector_size(self, collection_name):
        """
        Returns the size of the vectors in the collection.

        Args:
            collection_name (str): Name of the collection.

        Returns:
            int: The size of the vectors.

        Raises:
            CollectionNotFoundError: If the specified collection does not exist.
        """

        if collection_name not in self.collections:
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")
        collection = self.collections[collection_name]
        if not collection["documents"]:
            return 0
        return len(collection["documents"][0]['vector'])

    def get_collection_info(self, collection_name):
        """
        Provides high-level information about a collection.

        Args:
            collection_name (str): Name of the collection.

        Returns:
            dict: A dictionary containing collection information.

        Raises:
            CollectionNotFoundError: If the specified collection does not exist.
        """

        if collection_name not in self.collections:
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")
        collection = self.collections[collection_name]
        return {
            "document_count": len(collection["documents"]),
            "vector_size": self.get_vector_size(collection_name),
            "metadata": collection["metadata"],
        }

    def search(self, query, collection_name, top_k=5, metric="cosine"):
        """
        Searches a collection for the top-k most relevant documents.

        Args:
            query (str): The search query.
            collection_name (str): Name of the collection to search.
            top_k (int, optional): Number of top documents to return. Defaults to 5.
            metric (str, optional): Similarity metric ("cosine", "euclidean", "manhattan"). Defaults to "cosine".

        Returns:
            list: A list of dictionaries containing scores and documents.

        Raises:
            CollectionNotFoundError: If the specified collection does not exist.
            ValueError: If the similarity metric is unsupported.
        """
        if collection_name not in self.collections:
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")
        collection = self.collections[collection_name]
        query_vector = self._generate_embedding(query)
        distances = []

        for doc in collection["documents"]:
            doc_vector = doc['vector']
            score = self._calculate_similarity(query_vector, doc_vector, metric)
            distances.append((score, doc))

        distances.sort(key=lambda x: x[0], reverse=True if metric == "cosine" else False)
        return [{"score": round(dist[0], 4), "document": dist[1]} for dist in distances[:top_k]]

    def _calculate_similarity(self, query_vector, doc_vector, metric):
        """
            Calculates similarity between the query vector and document vector.

            Args:
                query_vector (list): The vector representing the query.
                doc_vector (list): The vector representing the document.
                metric (str): The similarity metric ("cosine", "euclidean", "manhattan").

            Returns:
                float: The similarity score.

            Raises:
                ValueError: If the similarity metric is unsupported.
        """
        
        if metric == "cosine":
            return self._cosine_similarity(query_vector, doc_vector)
        elif metric == "euclidean":
            return self._euclidean_distance(query_vector, doc_vector)
        elif metric == "manhattan":
            return self._manhattan_distance(query_vector, doc_vector)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def _euclidean_distance(self, vector1, vector2):
        """
        Calculates Euclidean distance between two vectors.
        """
        return np.linalg.norm(np.array(vector1) - np.array(vector2))

    def _cosine_similarity(self, vector1, vector2):
        """
        Calculates Cosine similarity between two vectors.
        """
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        return dot_product / (norm1 * norm2)

    def _manhattan_distance(self, vector1, vector2):
        """
        Calculates Manhattan distance between two vectors.
        """
        return np.sum(np.abs(np.array(vector1) - np.array(vector2)))

    def get_document(self, collection_name, document_id):
        """
        Retrieves a document by its ID from a collection.

        Args:
            collection_name (str): Name of the collection.
            document_id (int): ID of the document.

        Returns:
            dict: The document details.

        Raises:
            CollectionNotFoundError: If the specified collection does not exist.
            DocumentNotFoundError: If the document ID is not found in the collection.
        """

        if collection_name not in self.collections:
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")
        doc = next((d for d in self.collections[collection_name]['documents'] if d['id'] == document_id), None)
        if not doc:
            raise DocumentNotFoundError(f"Document with ID '{document_id}' not found.")
        return doc

    def get_collections(self):
        """
        Lists all collections in the database.

        Returns:
            list: A list of collection names.
        """
         
        return list(self.collections.keys())

    def get_all_docs(self, collection_name):
        """
        Retrieves all documents from a specified collection.

        Args:
            collection_name (str): Name of the collection.

        Returns:
            dict: A dictionary containing all documents and their details.

        Raises:
            CollectionNotFoundError: If the specified collection does not exist.
        """

        if collection_name not in self.collections:
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")

        collection = self.collections[collection_name]
        documents = collection['documents']

        if not documents:
            return {"error": f"No documents found in collection '{collection_name}'."}

        docs_data = {
            "collection_name": collection_name,
            "document_count": len(documents),
            "vector_size": self.get_vector_size(collection_name),
            "documents": []
        }

        sorted_documents = sorted(documents, key=lambda doc: doc['id'])

        for doc in sorted_documents:
            doc_info = {
                "id": doc['id'],
                "content": doc['content'],
                "vector": doc['vector'],
                "metadata": doc.get('metadata', 'No metadata available'),
            }
            docs_data["documents"].append(doc_info)

        return docs_data
