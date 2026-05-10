import polars as pl
import os
import sklearn
from sklearn.datasets import load_diabetes

pl.Config(set_tbl_cols=15,
          set_tbl_rows=100)

def get_diabetes_data() -> pl.LazyFrame:
    # Get sample data
    diabetes_data = load_diabetes()
    # Show info about data and columns
    print(diabetes_data['DESCR'])
    print(f'Sample size: {(sample_size := len(diabetes_data['data']))}')
    # Convert data to LF and assign column names
    diabetes_lf = (
        pl.LazyFrame(
            data=diabetes_data['data'],
            schema=diabetes_data['feature_names']
        )
        .rename(
            {
            's1': 'tc',
            's2': 'ldl',
            's3': 'hdl', 
            's4': 'tch',
            's5': 'ltg',
            's6': 'glu'
            }
        )
        .with_columns(
            # Apply some transformations to provided features (this ratio has little meaning as the provided columns are standardized)
            (pl.col('ldl')/pl.col('hdl')).alias('ldl_to_hdl_ratio'),
            # Add the target/y variable to be predicted
            pl.Series('target', diabetes_data['target']),
                      )
        .with_columns(
            # Center and divide by std so this column matches others
            ((pl.col("ldl_to_hdl_ratio") - pl.col("ldl_to_hdl_ratio").mean()
              )/(pl.col("ldl_to_hdl_ratio").std()*(sample_size**0.5))
             ).alias('ldl_to_hdl_ratio')
        )
    )
    return diabetes_lf

if __name__ == "__main__":
    print(get_diabetes_data().collect())