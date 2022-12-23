import pandas as pd
from conf.conf import logging
"""
Data extraction from scv file
""" 
def get_data(link: str) -> pd.DataFrame:
    logging.info("Reading data")
    df = pd.read_csv(link)

    return df 
