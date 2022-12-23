import pandas as pd

from connector.scv_connector import get_data
from sklearn.linear_model import LinearRegression
from util.model import save_model, load_model
from util.df import encode_df, split
from conf.conf import settings, logging

def training(train_ohe: list) -> None: 
    x_train_ohe = train_ohe.drop('Profit', axis=1)
    y_train_ohe = train_ohe['Profit']

    logging.info("Initialising LinearRegression")
    LinearRegressionModelInitializer = LinearRegression()
    logging.info("Training LinearRegression")
    LinearRegressionModel = LinearRegressionModelInitializer.fit(x_train_ohe, y_train_ohe)
    
    save_model(settings.linear_regression.MODEL_DIR, LinearRegressionModel)

def prediction(test_ohe: list) -> None:
    x_test_ohe = test_ohe.drop('Profit', axis=1)
    y_test_ohe = test_ohe['Profit']

    logging.info("Loading LR model")
    pickled_model = load_model(settings.linear_regression.MODEL_DIR)
    logging.info("Predicting LR")
    predictions_linear_regression_ohe = pickled_model.predict(x_test_ohe)

    logging.info(f'LR prediction is {predictions_linear_regression_ohe}')
    output_linear_regression_ohe = pd.DataFrame({ 'realProfit': y_test_ohe, 'Profit': predictions_linear_regression_ohe})
    
    return output_linear_regression_ohe

def process():
    df = get_data(settings.data.DATA_SET)
    df = encode_df(df)
    train_ohe, test_ohe = split(df, settings.general.DF_SPLIT_TEST_SIZE, settings.general.DF_SPLIT_RANDOM_STATE)
    clf = training(train_ohe)
    prediction_result = prediction(test_ohe)
    return prediction_result
if __name__ == '__main__':
    process()