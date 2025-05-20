import pandas as pd
import numpy as np
import os

def load_data_from_pandas_df(df, protected_attribute=None, binary_dict=None, id_attribute=None, order_by=None, ascending=False):
    """
    Loads data from a pandas DataFrame, optionally processing a protected attribute, sorting, and extracting IDs.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        protected_attribute (str, optional): The column name of the protected attribute to process. Defaults to None.
        binary_dict (dict, optional): A dictionary to map values in the protected attribute column to binary values (0 and 1). Defaults to None.
        id_attribute (str, optional): The column name to use as unique identifiers. If None, uses the DataFrame index. Defaults to None.
        order_by (str or list of str, optional): Column(s) to sort the DataFrame by. Defaults to None.
        ascending (bool, optional): Whether to sort in ascending order. Defaults to False.

    Returns:
        tuple: 
            - protected_values (np.ndarray): The values of the ranked protected attribute in 0/1 format. 
            - ids (np.ndarray): The array of IDs corresponding to each row (from id_attribute or DataFrame index).
    """

    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert protected_attribute is None or protected_attribute in df.columns, "Protected attribute must be a column in the DataFrame"
    assert id_attribute is None or id_attribute in df.columns, "ID attribute must be a column in the DataFrame"
    assert order_by is None or order_by in df.columns, "Order by attribute must be a column in the DataFrame"
    assert ascending in [True, False], "Ascending must be a boolean value"
    assert binary_dict is None or (isinstance(binary_dict, dict) and all(v in [0, 1] for v in binary_dict.values())), "Binary dictionary must be a dictionary with values 0 or 1"

    df = df.copy()
    if binary_dict is not None:
        df[protected_attribute] = df[protected_attribute].map(pd.Series(binary_dict))
    if order_by is not None:
        df = df.sort_values(by=order_by, ascending=ascending)
    if id_attribute is not None:
        ids = df[id_attribute].values
    else:
        ids = df.index.values
    df.dropna(inplace=True)
    if protected_attribute is not None:
        protected_values = df[protected_attribute].values
    else: 
        protected_values = df.values.flatten()
    return protected_values, ids


def load_data_from_list(data_list, id_list=None):
    """
    Loads data from a list, optionally extracting IDs.

    Parameters:
        data_list (list or np.ndarray): The input list or array containing the ranked binary values (0/1).
        id_list (list or np.ndarray, optional): The list of IDs corresponding to each item in data_list. If None, uses the index of data_list. Defaults to None.

    Returns:
        protected_values (np.ndarray): The values of the ranked protected attribute in 0/1 format. 
        ids (np.ndarray): The array of IDs corresponding to each element in data_list (from id_list or index of data_list).
    """

    assert isinstance(data_list, (list, np.ndarray)), "Input must be a list or numpy array"
    assert all([v in [0, 1] for v in data_list]), "Data list must contain only binary values (0 or 1)"
    assert id_list is None or isinstance(id_list, (list, np.ndarray)), "ID list must be a list or numpy array"
    assert id_list is None or len(data_list) == len(id_list), "Data list and ID list must have the same length"

    protected_values = np.array(data_list)
    if id_list is not None:
        ids = np.array(id_list)
    else:
        ids = np.arange(len(data_list))
    return protected_values, ids


def load_from_csv(file_path, protected_attribute=None, binary_dict=None, id_attribute=None, order_by=None, ascending=False):
    """
    Loads data from a CSV file, optionally processing a protected attribute, sorting, and extracting IDs.

    Parameters:
        file_path (str): The path to the CSV file.
        protected_attribute (str, optional): The column name of the protected attribute to process. Defaults to None.
        binary_dict (dict, optional): A dictionary to map values in the protected attribute column to binary values (0 and 1). Defaults to None.
        id_attribute (str, optional): The column name to use as unique identifiers. If None, uses the DataFrame index. Defaults to None.
        order_by (str or list of str, optional): Column(s) to sort the DataFrame by. Defaults to None.
        ascending (bool, optional): Whether to sort in ascending order. Defaults to False.

    Returns:
        tuple: 
            - protected_values (np.ndarray): The values of the ranked protected attribute in 0/1 format. 
            - ids (np.ndarray): The array of IDs corresponding to each row (from id_attribute or DataFrame index).
    """

    assert isinstance(file_path, str), "File path must be a string"
    assert os.path.isfile(file_path), "File does not exist"
    assert protected_attribute is None or isinstance(protected_attribute, str), "Protected attribute must be a string"
    assert id_attribute is None or isinstance(id_attribute, str), "ID attribute must be a string"
    assert order_by is None or isinstance(order_by, str) or isinstance(order_by, list), "Order by attribute must be a string or list of strings"
    assert ascending in [True, False], "Ascending must be a boolean value"
    assert binary_dict is None or (isinstance(binary_dict, dict) and all(v in [0, 1] for v in binary_dict.values())), "Binary dictionary must be a dictionary with values 0 or 1"

    df = pd.read_csv(file_path)
    return load_data_from_pandas_df(df, protected_attribute=protected_attribute, binary_dict=binary_dict,
                                     id_attribute=id_attribute, order_by=order_by, ascending=ascending)


def load_from_excel(file_path, sheet_name=0, protected_attribute=None, binary_dict=None, id_attribute=None, order_by=None, ascending=False):
    """
    Loads data from an Excel file, optionally processing a protected attribute, sorting, and extracting IDs.

    Parameters:
        file_path (str): The path to the Excel file.
        sheet_name (str or int, optional): The name or index of the sheet to load. Defaults to 0 (first sheet).
        protected_attribute (str, optional): The column name of the protected attribute to process. Defaults to None.
        binary_dict (dict, optional): A dictionary to map values in the protected attribute column to binary values (0 and 1). Defaults to None.
        id_attribute (str, optional): The column name to use as unique identifiers. If None, uses the DataFrame index. Defaults to None.
        order_by (str or list of str, optional): Column(s) to sort the DataFrame by. Defaults to None.
        ascending (bool, optional): Whether to sort in ascending order. Defaults to False.

    Returns:
        tuple: 
            - protected_values (np.ndarray): The values of the ranked protected attribute in 0/1 format. 
            - ids (np.ndarray): The array of IDs corresponding to each row (from id_attribute or DataFrame index).
    """

    assert isinstance(file_path, str), "File path must be a string"
    assert os.path.isfile(file_path), "File does not exist"
    assert protected_attribute is None or isinstance(protected_attribute, str), "Protected attribute must be a string"
    assert id_attribute is None or isinstance(id_attribute, str), "ID attribute must be a string"
    assert order_by is None or isinstance(order_by, str) or isinstance(order_by, list), "Order by attribute must be a string or list of strings"
    assert ascending in [True, False], "Ascending must be a boolean value"
    assert binary_dict is None or (isinstance(binary_dict, dict) and all(v in [0, 1] for v in binary_dict.values())), "Binary dictionary must be a dictionary with values 0 or 1"

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return load_data_from_pandas_df(df, protected_attribute=protected_attribute, binary_dict=binary_dict,
                                     id_attribute=id_attribute, order_by=order_by, ascending=ascending)


