import polars as pl
from sklearn.datasets import load_diabetes

pl.Config(set_tbl_cols=15,
          set_tbl_rows=100)

def get_diabetes_data() -> pl.LazyFrame:
    """
    Load diabetes data and apply basic transformations
    """
    # Get sample data
    diabetes_data = load_diabetes()
    # Show info about data and columns
    print(diabetes_data['DESCR'])
    print(f'Sample size: {(sample_size := len(diabetes_data['data']))}')

    def center_and_scale(column_name='ldl_to_hdl_ratio', 
                         sample_size=sample_size
                         ) -> pl.Expr:
        """
        Function for applying the same transformation used for the original columns to any new columns
        """
        return (
            # Center and divide by std so this column matches others
            ((pl.col(column_name) - pl.col(column_name).mean()
              )/(pl.col(column_name).std()*(sample_size**0.5))
             ).alias(column_name)
        )

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
            (pl.col('age')*pl.col('bmi')).alias('age_x_bmi_interaction'),
            # Add the target/y variable to be predicted
            pl.Series('target', diabetes_data['target'])
            )
        .with_columns(
            center_and_scale('ldl_to_hdl_ratio'),
            center_and_scale('age_x_bmi_interaction')
        )
    )
    return diabetes_lf

if __name__ == "__main__":
    print(get_diabetes_data().collect())
    print(get_diabetes_data().select(pl.col('ldl_to_hdl_ratio')**2,
                                     pl.col('age_x_bmi_interaction')**2).sum().collect())