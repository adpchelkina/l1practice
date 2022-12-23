import warnings
import argparse

from conf.conf import settings, logging
from util.metrics import rmse
from models.linear_regression.linear_regression_model import process as lr_process
from models.ridge.ridge_model import process as ridge_process
from models.lasso.lasso_model import process as lasso_process
warnings.filterwarnings("ignore")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='rmse metrics for LR, Lasso and Ridge')

  parser.add_argument('is_lr', choices=('True','False'), help='Calculate metrics for LR')
  parser.add_argument('is_lasso', choices=('True','False'), help='Calculate metrics for Lasso')
  parser.add_argument('is_ridge', choices=('True','False'), help='Calculate metrics for Ridge')
  args = parser.parse_args()


  if args.is_lr == 'True':
    logging.info('Processing linear regression model')
    output_linear_regression_ohe =lr_process()
    logging.info('Linear regression model procession has been finished')
  if args.is_ridge == 'True':
    logging.info('Processing ridge model')
    output_ridge_ohe = ridge_process()
    logging.info('Ridge model procession has been finished')
  if args.is_lasso == 'True':
    logging.info('Processing lasso model')
    output_lasso_ohe = lasso_process()
    logging.info('Linear lasso has been finished')

  logging.info('Calculating rmse for all models')
  if args.is_lr== 'True':
    linear_regression_ohe_rmse = rmse(output_linear_regression_ohe, output_linear_regression_ohe)
    logging.info(f'rmse linear_regression_ohe_rmse = {linear_regression_ohe_rmse}')
  if args.is_ridge== 'True':
    ridge_ohe_rmse = rmse(output_ridge_ohe, output_ridge_ohe)
    logging.info(f'rmse ridge_ohe_rmse = {ridge_ohe_rmse}')

  if args.is_lasso== 'True':
    lasso_ohe_rmse = rmse(output_lasso_ohe, output_lasso_ohe)
    logging.info(f'rmse lasso_ohe_rmse = {lasso_ohe_rmse}')

