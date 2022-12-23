import pandas as pd

def rmse(y: pd.DataFrame, output: pd.DataFrame) -> float:
  squares = (y['realProfit']-output['Profit'])**2

  return (squares.sum()/len(squares.index))**(1/2)