class CollectionNotFoundError(Exception):
    """Raised when a collection is not found in the database."""
    pass

class DatabaseLoadError(Exception):
    """Raised when there is an error loading the database."""
    pass

class DatabaseSaveError(Exception):
    """Raised when there is an error saving the database."""
    pass

class CollectionAlreadyExistsError(Exception):
    """Raised when trying to create a collection that already exists."""
    pass

class EmbeddingModelNotSetError(Exception):
    """Raised when the embedding model is not set."""
    pass

class DocumentNotFoundError(Exception):
    """Raised when a document is not found in a collection."""
    pass

class DocumentAlreadyExistsError(Exception):
    """Raised when trying to add a document that already exists."""
    pass


