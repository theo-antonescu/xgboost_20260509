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
    # Convert data to LF and assign column names
    diabetes_lf = (
        pl.LazyFrame(
            data=diabetes_data['data'],
            schema=diabetes_data['feature_names']
        )
        # Add the target/y variable to be predicted
        .with_columns(pl.Series('target', diabetes_data['target']))
    )
    return diabetes_lf

if __name__ == "__main__":
    print(get_diabetes_data().collect())
    
