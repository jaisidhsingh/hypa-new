import numpy as np

def cosine_distance(embeddings1, embeddings2):
    """
    Calculate the cosine distance between two sets of embeddings.
    
    Args:
    embeddings1 (np.ndarray): First set of embeddings with shape (n_samples, n_features)
    embeddings2 (np.ndarray): Second set of embeddings with shape (n_samples, n_features)
    
    Returns:
    np.ndarray: Cosine distance between each pair of embeddings
    """
    # Normalize the embeddings
    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    normalized_embeddings1 = embeddings1 / norm1
    normalized_embeddings2 = embeddings2 / norm2
    
    # Calculate cosine similarity
    cosine_similarity = np.dot(normalized_embeddings1, normalized_embeddings2.T)
    
    # Convert similarity to distance
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance
