# Dimensia - Vector Database for Document Embeddings and Semantic Search

Dimensia is a lightweight vector database designed for managing and querying documents using embeddings. It supports creating collections, adding documents with optional metadata, performing semantic searches, and retrieving detailed document information.

## Features

- **Embeddings**: Generates document embeddings using the `sentence-transformers` library, providing vector representations of text.
- **Collections**: Allows you to create, manage, and query collections of documents.
- **Search**: Performs efficient semantic searches for the top-k most relevant documents based on a query, using cosine, Euclidean, or Manhattan distance metrics.
- **Concurrency**: Uses thread pooling to handle document ingestion concurrently for efficient processing.
- **Metadata**: Supports storing metadata with documents for enriched queries.
- **Duplicate Detection**: Identifies and skips duplicate documents based on content hashes.

## Installation

You can install Dimensia via `pip`:

```bash
pip install dimensia
``` 

## Usage

```python

from dimensia import Dimensia

# Initialize the database
db = Dimensia(db_path="dimensia_db")
print("Database initialized.")

# Set the embedding model (example: a transformer model for NLP tasks)
embedding_model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
db.set_embedding_model(embedding_model_name)
print(f"Embedding model '{embedding_model_name}' set successfully.")

# prepare documents to ingest
documents = [
    {"content": "The advancements in deep learning have revolutionized AI applications.", "metadata": {}},
    {"content": "Natural Language Processing models are increasingly effective in understanding text.", "metadata": {}},
    {"content": "Recent research shows that transformers outperform traditional neural networks in NLP.", "metadata": {}},
    {"content": "Machine learning models are being used in healthcare for predictive analysis.", "metadata": {}},
    {"content": "Reinforcement learning is transforming robotics and autonomous systems.", "metadata": {}},
]

# Define collection name for research articles
collection_name = "research_articles"

# Check if collection exists, create it if not
if collection_name not in db.get_collections():
    db.create_collection(collection_name)
    print(f"Collection '{collection_name}' created successfully.")
else:
    print(f"Collection '{collection_name}' already exists.")

# Add documents (research articles) to the collection with metadata
db.add_documents(collection_name, documents)
print(f"Documents added to collection '{collection_name}' successfully.")

# Retrieve and print collection information (number of documents, vector size, etc.)
info = db.get_collection_info(collection_name)
print(f"Collection info for '{collection_name}': {info}")

# Get structure of the collection (document count, vector size)
structure = db.get_structure(collection_name)
print(f"Structure for collection '{collection_name}': {structure}")

# Retrieve vector size (size of embeddings) for the collection
vector_size = db.get_vector_size(collection_name)
print(f"Vector size for collection '{collection_name}': {vector_size}")

# Search for relevant research articles using a query
query = "How transformers are applied in NLP"
results = db.search(query, collection_name, top_k=3, metric="cosine")
print(f"Search results for query '{query}':")
for result in results:
    print(f"Score: {result['score']}, Document: {result['document']['content']}")

# Retrieve a specific document by ID (e.g., ID 1)
document_id = 1
document = db.get_document(collection_name, document_id)
print(f"Retrieved document with ID {document_id}: {document}")

# Get all documents in the collection, sorted by document ID
all_docs = db.get_all_docs(collection_name)
print(f"All documents in '{collection_name}':")
for doc in all_docs["documents"]:
    print(f"ID: {doc['id']}, Content: {doc['content']}, Metadata: {doc['metadata']}")

```

## Contributing

We welcome contributions to improve **Dimensia**! Please fork the repository from [Dimensia GitHub Repository](https://github.com/aniruddhasalve/dimensia), make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Support

If you encounter any issues or have questions, please don't hesitate to open an issue on our [GitHub issues page](https://github.com/aniruddhasalve/dimensia/issues). We welcome feedback, bug reports, and feature requests!

We strive to respond as quickly as possible to all issues and questions.