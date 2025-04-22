import pandas as pd
from neural_network_lib.utils import clean_dataset
from neural_network_lib.utils import train_test_split
from colorama import Fore, Style
import os

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a label column 'diagnosis' to the dataset based on some condition or rule.
    
    Args:
    df : DataFrame
        The input data to label.
        
    Returns:
    df : DataFrame
        The labeled data.
    """
    column_names = [
        'radius', 'texture', 'perimeter', 'area', 'smoothness', 
        'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension',
    ]
    new_columns = ['id', 'diagnosis'] + [column_names[i % 10] + '_' + ['mean', 'se', 'worst'][i // 10] for i in range(df.columns.size - 2)]
    df.columns = new_columns

    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    return df

def split_dataset(input_csv: str, test_size=0.2, random_state=32):
    """
    Reads a CSV file, labels the data, splits the data into train and test sets, and saves them to CSV files.
    
    Args:
        input_csv : str
            The path to the input CSV file.
        test_size : float, optional
            The proportion of the dataset to include in the test split. Default is 0.2.
        random_state : int, optional
            Controls the shuffling applied to the data before applying the split. Default is 32.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        ValueError: If the input CSV file is empty or if there are issues with the data processing.

    Returns:
        None
    """
    print(f"{Fore.BLUE}Training parameters:{Style.RESET_ALL}")
    print(f"  - Input csv: {input_csv}")
    print(f"  - Test size: {test_size}")
    print(f"  - Random State: {random_state}")

    try:
        data = pd.read_csv(input_csv)
    except FileNotFoundError:
        raise FileNotFoundError(f'{Fore.RED}Input file {input_csv} not found')
    
    if data.empty:
        raise ValueError(f'{Fore.RED}Input file {input_csv} is empty')

    try:
        data, operations_log = clean_dataset(data)
        
        if any(operations_log.values()):
            print("Cleaning operations performed:")
            for operation, count in operations_log.items():
                if count > 0:
                    print(f"{operation}: {count} times\n")

        data = add_labels(data)

        X = data.drop(columns=['diagnosis'])
        y = data['diagnosis']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        os.makedirs('data/train_test', exist_ok=True)
        train_data.to_csv('data/train_test/train.csv', index=False)
        test_data.to_csv('data/train_test/test.csv', index=False)
        
        print(f'{Fore.CYAN}Training data saved to data/train_test/train.csv{Style.RESET_ALL}')
        print(f'{Fore.CYAN}Validation data saved to data/train_test/test.csv{Style.RESET_ALL}\n')
    except Exception as e:
        raise ValueError(f'Error processing data: {str(e)}')