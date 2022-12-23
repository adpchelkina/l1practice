import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from conf.conf import settings, logging

"""
Splitting data from dataframe
""" 
def split(data_ohe: pd.DataFrame, test_size:float, random_state: int) -> Tuple[list, list]:
    logging.info(f"Splitting {data_ohe}")
    train_ohe, test_ohe = train_test_split(
        data_ohe, 
        test_size=test_size, 
        random_state=random_state
    )

    return train_ohe, test_ohe

"""
encodes data in appropriate format
"""
def encode_df(df: pd.DataFrame) -> list:
    logging.info(f"Encoding df")
    df = df[(np.abs(stats.zscore(df['Profit'])) < 3)]
    df = df.reset_index(drop=True)
    df = df.drop(['Country'], axis=1)

    categorical_deatures = settings.data_categories.CATEGORICAL_DEATURES
    data_ohe = df
    for i in categorical_deatures:
        data_ohe = make_one_hot_encoding(data_ohe, i)

    return data_ohe

"""
one itaration process of encoding
"""
def make_one_hot_encoding(in_data: pd.DataFrame, col_name: str) ->  pd.DataFrame:
    encoder = OneHotEncoder(handle_unknown='ignore', drop = 'first')

    encoder_df = pd.DataFrame(encoder.fit_transform(in_data[[col_name]]).toarray())
    column_name = encoder.get_feature_names_out([col_name])

    encoder_df.columns = column_name

    in_data = in_data.join(encoder_df)
    in_data = in_data.drop([col_name], axis=1)

    return in_data