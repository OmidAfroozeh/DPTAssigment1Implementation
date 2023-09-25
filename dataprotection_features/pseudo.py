"""
This modules implements a pseudonymization method.
"""
import pandas as pd
import random
import hashlib

def pseudonymize_column(df, column_name):
    """
    Pseudonymize a column in a DataFrame using a mapping approach.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data to be pseudonymized.
        column_name (str): The name of the column to be pseudonymized.

    Returns:
        pandas.DataFrame: A new DataFrame with the specified column pseudonymized.

    This function replaces the values in the specified column with pseudonyms generated using a mapping approach.
    It maintains a mapping of original values to pseudonyms to allow for reversibility.

    """
    # Make a copy of the input DataFrame to avoid modifying the original data
    pseudonymized_df = df.copy()

    # Create a mapping dictionary to store original values and pseudonyms
    mapping = {}

    # Generate pseudonyms for each unique value in the column
    unique_values = df[column_name].unique()
    for value in unique_values:
        # Generate a pseudonym using a hash function (SHA-256 in this case)
        pseudonym = hashlib.sha256(str(value).encode()).hexdigest()

        # Store the mapping of the original value to the pseudonym
        mapping[value] = pseudonym

    # Replace the column values with pseudonyms using the mapping
    pseudonymized_df[column_name] = df[column_name].replace(mapping)

    return (mapping,pseudonymized_df)

