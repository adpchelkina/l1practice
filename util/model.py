import pickle
from conf.conf import logging

def save_model(dir: str, model) -> None:
    logging.info(f"Saving model")
    pickle.dump(model, open(dir, 'wb'))
    
def load_model(dir: str):
    return pickle.load(open(dir, 'rb'))
    
