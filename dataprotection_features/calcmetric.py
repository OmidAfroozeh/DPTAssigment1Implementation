"""
This module contains functions to measure several privacy metrics such as k-anonymity and l-diversity
"""
import pandas as pd

def satisfies_k_anonymity(df, k, qi_columns):
    """
    Determine if a dataset satisfies k-anonymity based on quasi-identifier columns.

    This function checks whether the input DataFrame satisfies k-anonymity by grouping the data based on
    the quasi-identifier columns and making sure that no group has fewer than k records.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing sensitive and quasi-identifier data.
        k (int): The desired k value for k-anonymity (minimum number of records in each equivalent group).
        qi_columns (list): A list of column names representing the quasi-identifier attributes.

    Returns:
        bool: True if the DataFrame satisfies k-anonymity, False otherwise.
    """
    # Group the DataFrame by the quasi-identifier columns
    grouped = df.groupby(qi_columns)

    # Check if any group has fewer than k records
    for group_name, group_data in grouped:
        if len(group_data) < k:
            print(f"Dataset does not satisfy {k}-anonymity")
            return False
    print(f"Dataset satisfies {k}-anonymity")
    return True

def find_k_anonymity(df, qi_columns):
    """
    Find the maximum k value for which k-anonymity is satisfied in the entire dataset.

    This function iteratively increases the value k until the dataset no longer satisfies k-anonymity.
    It uses the 'satisfies_k_anonymity' function to check if the dataset satisfies k-anonymity or not.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing sensitive and quasi-identifier data.
        qi_columns (list): A list of column names representing the quasi-identifier attributes.

    Returns:
        int: The maximum k for which the dataset still satisfies k-anonymity.
    """
    k = 1
    while satisfies_k_anonymity(df, k, qi_columns):
        k += 1
    print(f"Dataset satisfies maximum {k-1}-anonymity")
    return k - 1

def satisfies_l_diversity(df, l, qi_columns, sensitive_column):
    """
    Determine if a dataset satisfies l-diversity based on quasi-identifier columns and a sensitive column.

    This function checks if the input DataFrame satisfies l-diversity by grouping the data based on
    the specified quasi-identifier columns and ensuring that each group has at least 'l' unique values
    in the sensitive column.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing sensitive and quasi-identifier data.
        l (int): The desired l value for l-diversity (minimum number of unique values in the sensitive column).
        qi_columns (list): A list of column names representing the quasi-identifier attributes.
        sensitive_column (str): The name of the column containing sensitive information.

    Returns:
        bool: True if the DataFrame satisfies l-diversity, False otherwise.
    """
    # Group the DataFrame by the quasi-identifiers
    groups = df.groupby(qi_columns)

    # Check if any group has fewer unique values in the sensitive column than 'l'
    for group_name, group_data in groups:
        sensitive_counts = group_data[sensitive_column].nunique()
        if sensitive_counts < l:
            print(f"Dataset does not satisfy {l}-diversity")
            return False

    # If all groups have at least 'l' unique values, the dataset satisfies l-diversity
    print(f"Dataset satisfies {l}-diversity")
    return True

def find_l_diversity(df, qi_columns, sensitive_column):
    """
    Find the maximum l value for which l-diversity is satisfied in the entire dataset.

    This function iteratively increases the value 'l' until the dataset no longer satisfies l-diversity.
    It uses the 'satisfies_l_diversity' function to check if the dataset satisfies l-diversity or not.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing sensitive and quasi-identifier data.
        qi_columns (list): A list of column names representing the quasi-identifier attributes.
        sensitive_column (str): The name of the column containing sensitive information.

    Returns:
        int: The maximum 'l' for which the dataset still satisfies l-diversity.
    """
    l = 1
    while satisfies_l_diversity(df, l, qi_columns, sensitive_column):
        l += 1
    print(f"Dataset satisfies maximum {l-1}-diversity")
    return l - 1
