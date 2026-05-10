import polars as pl
import xgboost as xgb
from get_training_data import get_diabetes_data
from sklearn.model_selection import train_test_split

diabetes_data = get_diabetes_data()
train_data, test_data = train_test_split(diabetes_data.collect(), 
                                   test_size=0.15,
                                   random_state=123)

train_y = train_data.select(pl.col('target'))
test_y = test_data.select(pl.col('target'))

print(train_data, test_data)

xgb_regressor = xgb.XGBRegressor(tree_method='hist', max_depth=10, 
                                 gamma=100, eta=0.1, 
                                 device='cpu')
xgb_regressor.fit(
    X=train_data.drop(['target', 'ldl_to_hdl_ratio']), 
    y=train_y,
    eval_set=[(test_data.drop(['target', 'ldl_to_hdl_ratio']), test_y)]
    )

test_y.with_columns(pl.Series('predicted_y', xgb_regressor.predict(test_data.drop(['target', 'ldl_to_hdl_ratio']))))
