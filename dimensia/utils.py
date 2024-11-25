import os

def ensure_directory_exists(directory):
    """
    Ensures that the specified directory exists. If the directory does not exist, it creates the directory.

    Args:
        directory (str): The path to the directory to be checked/created.

    Returns:
        None: This function does not return anything. It either creates the directory or does nothing if it already exists.

    Raises:
        OSError: If there is an issue creating the directory (e.g., lack of permissions).
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
