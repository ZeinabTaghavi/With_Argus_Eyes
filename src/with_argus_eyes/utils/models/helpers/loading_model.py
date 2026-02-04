import pickle

def load_model(model_path):
    """
    Loads and returns a model from the given path.

    Args:
        model_path (str): Path to the model file.

    Returns:
        object: The loaded model.
    """
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
