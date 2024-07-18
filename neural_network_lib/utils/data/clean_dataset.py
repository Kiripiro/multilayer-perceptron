import pandas as pd

def clean_dataset(df: pd.DataFrame, dropna_threshold=0.5, dropna_rows=True, remove_duplicates=True):
    """
    Cleans the dataset by handling missing values, removing duplicates, and checking for data consistency.
    Logs the operations performed and their frequency.

    Args:
        df (pd.DataFrame): The input dataframe to be cleaned.
        dropna_threshold (float): Proportion of non-NA values required to keep a column. Default is 0.5.
        dropna_rows (bool): Whether to drop rows with any NA values. Default is True.
        remove_duplicates (bool): Whether to remove duplicate rows. Default is True.

    Returns:
        pd.DataFrame: The cleaned dataset.
        dict: A dictionary with the number of operations performed.
    """
    operations_log = {
        'dropna_columns': 0,
        'dropna_rows': 0,
        'remove_duplicates': 0
    }

    initial_columns = df.shape[1]
    df_cleaned = df.dropna(axis=1, thresh=int(dropna_threshold * len(df)))
    dropped_columns = initial_columns - df_cleaned.shape[1]
    if dropped_columns > 0:
        operations_log['dropna_columns'] = dropped_columns
    
    if dropna_rows:
        initial_rows = df_cleaned.shape[0]
        df_cleaned = df_cleaned.dropna(axis=0, how='any')
        dropped_rows = initial_rows - df_cleaned.shape[0]
        if dropped_rows > 0:
            operations_log['dropna_rows'] = dropped_rows
    
    if remove_duplicates:
        initial_rows = df_cleaned.shape[0]
        df_cleaned = df_cleaned.drop_duplicates()
        removed_duplicates = initial_rows - df_cleaned.shape[0]
        if removed_duplicates > 0:
            operations_log['remove_duplicates'] = removed_duplicates
    
    return df_cleaned, operations_log
