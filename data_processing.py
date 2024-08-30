import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# def load_dataset(dataset_name):
#     """
#     Load a dataset from a specified path based on the dataset name.
#
#     Parameters:
#     dataset_name (str): The name of the dataset to load.
#
#     Returns:
#     DataFrame: A pandas DataFrame containing the loaded data.
#     """
#     # Construct the file path
#     file_path = os.path.join('dataset', dataset_name, f'{dataset_name}.csv')
#
#     try:
#         # Load the dataset
#         data = pd.read_csv(file_path)
#         print(f"Dataset '{dataset_name}' loaded successfully.")
#         return data
#     except FileNotFoundError:
#         print(f"Error: The file '{file_path}' does not exist.")
#         return None
#     except Exception as e:
#         print(f"An error occurred while loading the dataset: {e}")
#         return None

def load_dataset(dataset_name):
    """
    Load a dataset from a specified path based on the dataset name.

    Parameters:
    dataset_name (str): The name of the dataset to load.

    Returns:
    DataFrame: A pandas DataFrame containing the loaded data.
    """
    if dataset_name == 'amazon_binary':
        # 数据文件的路径
        file_path = os.path.join('dataset', 'amazon', 'amazon_output.csv')

        try:
            # 读取 CSV 文件
            full_data = pd.read_csv(file_path)

            # 获取类别标签的列表，并排序
            labels = sorted(full_data.iloc[:, -1].unique().tolist())

            # 前25个类别的标签
            first_25_labels = labels[:25]

            # 更新最后一列中的标签，前25个类别为类别0，其余为类别1
            full_data.iloc[:, -1] = full_data.iloc[:, -1].apply(lambda label: 0 if label in first_25_labels else 1)
            print(f"Dataset '{dataset_name}' loaded and labels processed successfully.")
            return full_data
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' does not exist.")
            return None
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}")
            return None
    else:
        # Construct the file path for other datasets
        file_path = os.path.join('dataset', dataset_name, f'{dataset_name}.csv')

        try:
            # Load the dataset
            data = pd.read_csv(file_path)
            print(f"Dataset '{dataset_name}' loaded successfully.")
            return data
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' does not exist.")
            return None
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}")
            return None


def split_dataset(data, train_size=0.55, valid_size=0.35, test_size=0.1):
    """
    Split the dataset into training, validation, and test sets based on specified proportions.

    Parameters:
    data (DataFrame): The pandas DataFrame to split.
    train_size (float): The proportion of the dataset to include in the train split.
    valid_size (float): The proportion of the dataset to include in the validation split.
    test_size (float): The proportion of the dataset to include in the test split.

    Returns:
    tuple: A tuple containing three DataFrames (train_set, validation_set, test_set).
    """
    if abs((train_size + valid_size + test_size) - 1.0) > 1e-6:
        raise ValueError("The sum of train, validation, and test sizes must equal 1.")

    # First split to separate out the training set
    train_data, temp_data = train_test_split(data, train_size=train_size)

    # Adjust valid_size to account for the reduced number of samples in temp_data
    valid_size_adjusted = valid_size / (valid_size + test_size)

    # Second split to separate out the validation and test sets
    validation_data, test_data = train_test_split(temp_data, train_size=valid_size_adjusted)

    return train_data, validation_data, test_data



def encode_categorical_variables(df):
    """
    Encodes all categorical columns in the DataFrame using label encoding.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.

    Returns:
    pandas.DataFrame: The DataFrame with categorical variables encoded.
    """
    # Create a copy to avoid modifying the original data
    df_encoded = df.copy()

    # Instantiate LabelEncoder
    le = LabelEncoder()

    # Iterate through each column, checking data type
    for column in df_encoded.columns:
        # If the data type is 'object' (usually strings), we assume it is categorical
        if df_encoded[column].dtype == 'object' or df_encoded[column].dtype.name == 'category':
            # Apply Label Encoder to the categorical column
            df_encoded[column] = le.fit_transform(df_encoded[column])

    return df_encoded


def load_feature_names(dataset_name):
    """
    Load feature names from a text file within a specific dataset directory.

    Parameters:
    dataset_name (str): The name of the dataset whose feature names are to be loaded.

    Returns:
    list: A list of feature names.
    """
    # Construct the path to the features_name.txt file
    if dataset_name == 'amazon_binary':
        file_path = os.path.join('dataset', 'amazon', 'features_name.txt')
    else:
        file_path = os.path.join('dataset', dataset_name, 'features_name.txt')

    try:
        # Open the file and read the lines
        with open(file_path, 'r') as file:
            # Read all lines and strip any leading/trailing whitespace characters
            feature_names = file.read().strip().split(', ')
        print(f"Feature names loaded successfully from {file_path}.")
        return feature_names
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the feature names: {e}")
        return None

def load_operations():
    """
    Load a list of mathematical operations from a text file.

    Returns:
    list: A list of operations.
    """
    # Define the file path relative to the current script location
    file_path_unary = os.path.join('dataset', 'operations_unary.txt')
    file_path_binary = os.path.join('dataset', 'operations_binary.txt')
    try:
        # Open the file and read the contents
        with open(file_path_unary, 'r') as file:
            # Assuming the file contains a single line of operations separated by commas
            operations_unary = file.read().strip().split(', ')
        print(f"Operations loaded successfully from {file_path_unary}.")
        with open(file_path_binary, 'r') as file:
            # Assuming the file contains a single line of operations separated by commas
            operations_binary = file.read().strip().split(', ')
        print(f"Operations loaded successfully from {file_path_binary}.")
        return operations_unary, operations_binary
    except FileNotFoundError:
        print(f"Error: The file '{file_path_unary}'or'{file_path_binary}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the operations: {e}")
        return None

def list_to_string(feature_names, delimiter=','):
    """
    Convert a list of feature names into a single string with a specified delimiter.

    Parameters:
    feature_names (list): The list of feature names.
    delimiter (str): The delimiter to use between feature names in the returned string.

    Returns:
    str: A single string containing all feature names separated by the delimiter.
    """
    if feature_names:
        return delimiter.join(feature_names)
    else:
        return ""


def load_dataset_description(dataset_name):
    """
    Load a dataset description from a text file within a specified dataset directory.

    Parameters:
    dataset_name (str): The name of the dataset whose description is to be loaded.

    Returns:
    str: The content of the dataset description file.
    """
    # Construct the path to the features_description.txt file
    file_path = os.path.join('dataset', dataset_name, 'features_description.txt')

    try:
        # Open the file and read the content
        with open(file_path, 'r') as file:
            description = file.read()
        print(f"Dataset description loaded successfully from {file_path}.")
        return description
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the dataset description: {e}")
        return None

def read_dataset_description(dataset_name):
    """
    Reads the dataset description from a text file located at a specific path.

    Returns:
    str: The content of the dataset description file or None if any error occurs.
    """
    # Set the path to the file relative to the current script location
    file_path = os.path.join('dataset', dataset_name, 'features_description.txt')

    try:
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            dataset_description = file.read()
        print("Dataset description loaded successfully.")
        return dataset_description
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the dataset description: {e}")
        return None


def generate_operation_examples(feature_names, operation_unary, operation_binary):
    """
    Generates examples of operations on given feature names using specified unary and binary operations,
    ensuring that the 'delete' operation is included for the second feature.

    Parameters:
    feature_names (list of str): Names of features to perform operations on.
    operation_unary (list of str): List of unary operations.
    operation_binary (list of str): List of binary operations.

    Returns:
    str: A string containing examples of operations.
    """
    examples = []
    # Add the delete operation for the second feature
    examples.append(f"delete {feature_names[1]} (name: None)")

    # Check if there are at least two feature names for binary operations
    if len(feature_names) >= 2:
        # Generate binary operation examples
        count_binary = 0
        for op in operation_binary:
            if count_binary < 4:  # Limit to four examples including unary operations
                examples.append(f"{feature_names[0]} {op} {feature_names[1]} (name: {feature_names[0]}_{op}_{feature_names[1]})")
                count_binary += 1
            if count_binary == 4:
                break

    # Generate unary operation examples
    count_unary = 1  # Starts at 1 to account for the delete operation
    for op in operation_unary:
        if op != 'delete':  # Exclude the 'delete' operation since it's already added
            if count_unary < 4:  # Limit to four examples including binary operations
                examples.append(f"{op} {feature_names[0]} (name: {op}_{feature_names[0]})")
                count_unary += 1
            if count_unary == 4:
                break

    return "\n".join(examples)


def preprocess_data(dataset):
    # Get the column name of the last column in the DataFrame
    target_column = dataset.columns[-1]

    # Separate features and target variable
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]

    # Encode the target variable using label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Determine which columns are numeric and which are categorical
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Create transformers for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler())  # Standardization
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding
    ])

    # Combine all transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply preprocessing to features
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y_encoded, preprocessor
