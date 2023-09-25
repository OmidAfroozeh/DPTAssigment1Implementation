"""
This module implements k-anonymity, l-diversity, and t-closeness methods.
Currently, it only supports numerical values as quasi-identifiers.
"""

import pandas as pd
import random
from tabulate import tabulate

class Mondrian:
    """
    Mondrian class for data anonymization using the Mondrian algorithm.
    """

    def __init__(self, df, feature_columns, sensitive_column=None):
        """
        Initialize the Mondrian instance.

        Parameters:
            df (pandas.DataFrame): The input DataFrame.
            feature_columns (list): A list of column names containing the feature attributes.
            sensitive_column (str, optional): The name of the column containing sensitive information.
        """
        self.df = df
        self.feature_columns = feature_columns
        self.sensitive_column = sensitive_column

    def is_valid(self, partition, k=2, l=0, t=0.0):
        """
        Check if a partition satisfies k-anonymity, l-diversity, and t-closeness.

        Parameters:
            partition (pandas.Index): The partition to check.
            k (int): The desired k-anonymity value.
            l (int): The desired l-diversity value.
            t (float): The maximum allowed t-closeness distance.

        Returns:
            bool: True if the partition satisfies privacy requirements, False otherwise.
        """
        # k-anonymous
        if not is_k_anonymous(partition, k):
            return False
        # l-diverse
        if l > 0 and self.sensitive_column is not None:
            diverse = is_l_diverse(self.df, partition, self.sensitive_column, l)
            if not diverse:
                return False
        # t-closeness
        if t > 0.0 and self.sensitive_column is not None:
            global_freqs = get_global_freq(self.df, self.sensitive_column)
            close = is_t_close(self.df, partition, self.sensitive_column, global_freqs, t)
            if not close:
                return False
        return True

    def get_spans(self, partition, scale=None):
        """
        Calculate the spans of feature columns within a partition.

        Parameters:
            partition (pandas.Index): The partition to calculate spans for.
            scale (dict, optional): Scaling factors for feature columns.

        Returns:
            dict: A dictionary mapping feature column names to their spans.
        """
        spans = {}
        for column in self.feature_columns:
            # for categorical columns, the unique number of values in the column is considered as the span
            if self.df[column].dtype.name == "category":
                span = len(self.df[column][partition].unique())
            # for numerical columns, the difference between max and min is considered as the span of the column
            else:
                span = (
                    self.df[column][partition].max() - self.df[column][partition].min()
                )
            if scale is not None:
                span = span / scale[column]
            spans[column] = span
        return spans

    def split(self, column, partition):
        """
        Split a partition based on a feature column.

        Parameters:
            column (str): The name of the feature column to split on.
            partition (pandas.Index): The partition to split.

        Returns:
            tuple: Two partitions after the split.
        """
        dfp = self.df[column][partition]
        if dfp.dtype.name == "category":
            values = dfp.unique()
            lv = set(values[: len(values) // 2])
            rv = set(values[len(values) // 2 :])
            return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
        else:
            median = dfp.median()
            dfl = dfp.index[dfp < median]
            dfr = dfp.index[dfp >= median]
            return (dfl, dfr)

    def partition(self, k=3, l=0, t=0.0):
        """
        Partition the input dataset into k-anonymous partitions.

        Parameters:
            k (int): The desired k-anonymity value.
            l (int): The desired l-diversity value.
            t (float): The maximum allowed t-closeness distance.

        Returns:
            list: List of k-anonymous partitions.

        This function performs data anonymization using the Mondrian algorithm. It partitions the input dataset into
        'k'-anonymous partitions while optionally satisfying 'l'-diversity and applying 't'-closeness.
        """
        scale = self.get_spans(self.df.index)
        finished_partitions = []
        # the algorithm start with the entire dataset as one partition and will iteratively split it into smaller parts
        partitions = [self.df.index]
        while partitions:
            partition = partitions.pop(0)
            spans = self.get_spans(partition, scale)
            for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                lp, rp = self.split(column, partition)
                # checks in the left partition or the right partition satisfy the criteria
                if not self.is_valid(lp, k, l, t) or not self.is_valid(rp, k, l, t):
                    continue
                partitions.extend((lp, rp))
                break
            else:
                finished_partitions.append(partition)
        return finished_partitions


def is_k_anonymous(partition, k):
    """
    Check if a partition contains at least k entries.

    Parameters:
        partition (pandas.Index): The partition to check.
        k (int): The desired k-anonymity value.

    Returns:
        bool: True if the partition satisfies k-anonymity, False otherwise.
    """
    if len(partition) < k:
        return False
    return True

def is_l_diverse(df, partition, sensitive_column, l):
    """
    Check if a partition satisfies l-diversity by checking if the sensitive_column contains l distinct values.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        partition (pandas.Index): The partition to check.
        sensitive_column (str): The column name containing sensitive information.
        l (int): The desired l-diversity value.

    Returns:
        bool: True if the partition satisfies l-diversity, False otherwise.
    """
    diversity = len(df.loc[partition][sensitive_column].unique())
    return diversity >= l

def is_t_close(df, partition, sensitive_column, global_freqs, t):
    """
    Check if a partition satisfies t-closeness.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        partition (pandas.Index): The partition to check.
        sensitive_column (str): The column name containing sensitive information.
        global_freqs (dict): Global frequency distribution of sensitive values.

    Returns:
        bool: True if the maximum distance is less than t threshold, False otherwise.
    """
    total_count = float(len(partition))
    d_max = None
    group_counts = (
        df.loc[partition].groupby(sensitive_column)[sensitive_column].agg("count")
    )
    for value, count in group_counts.to_dict().items():
        p = count / total_count
        d = abs(p - global_freqs[value])
        if d_max is None or d > d_max:
            d_max = d
    return d_max <= t

def get_global_freq(df, sensitive_column):
    """
    Calculate the global frequency distribution of sensitive values.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        sensitive_column (str): The column name containing sensitive information.

    Returns:
        dict: A dictionary mapping sensitive values to their global frequencies.
    """
    global_freqs = {}
    total_count = float(len(df))
    group_counts = df.groupby(sensitive_column)[sensitive_column].agg("count")
    for value, count in group_counts.to_dict().items():
        p = count / total_count
        global_freqs[value] = p
    return global_freqs

def showcase(df, n, partitions):
    """
    Display random partitions of a DataFrame to show the result.

    Parameters:
        df (pandas.DataFrame): The input DataFrame to print partitions from.
        n (int): The number of random partitions to showcase.
        partitions (list of pandas.Index or int): The list of partition indices.

    This function selects n random partitions and prints them using the tabulate library.
    """
    print("Showcasing some of the anonymized partition:")
    randomlist = []
    for i in range(0, n):
        rand = random.randint(0, len(partitions) - 1)
        randomlist.append(rand)
    for x in randomlist:
        print("Partition", x)
        print(tabulate(df.iloc[partitions[x]], headers='keys', tablefmt='psql'))


def __anonymize(df, feature_columns, sensitive_column, k, l=0, t=0.0):
    """
    Anonymize a dataset using the Mondrian algorithm.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        feature_columns (list): A list of column names representing the feature attributes.
        sensitive_column (str): The name of the column containing sensitive information.
        k (int): The desired k-anonymity value.
        l (int): The desired l-diversity value (default is 0, indicating no l-diversity requirement).
        t (float): The maximum allowed t-closeness distance (default is 0.0, indicating no t-closeness requirement).

    Returns:
        pandas.DataFrame: Anonymized version of the input DataFrame.

    This function performs data anonymization using the Mondrian algorithm. It partitions the input dataset into
    'k'-anonymous partitions while optionally satisfying 'l'-diversity and applying 't'-closeness.

    The anonymization process involves partitioning and aggregation of data to ensure that each
    partition satisfies the specified privacy requirements.
    """
    print("=========Start anonymization process:")
    print("Partition the dataset:")
    mondrian = Mondrian(df, feature_columns, sensitive_column)
    partitions = mondrian.partition(k, l, t)
    print(len(partitions), "partitions created.")
    return aggregate(df, partitions, feature_columns, sensitive_column)


def anonymize_k_anonymity(df, feature_columns, sensitive_column, k):
    """
    A wrapper function passing k-anonymity parameters to the anonymize function

    Parameters:
        df (pandas.DataFrame): The input DataFrame to be anonymized.
        feature_columns (list): List of column names to be used as quasi-identifiers for k-anonymity.
        sensitive_column (str): The column name containing sensitive information.
        k (int): The desired k-value for k-anonymity.

    Returns:
        pandas.DataFrame: A new DataFrame with k-anonymized data.
    """
    return __anonymize(df, feature_columns, sensitive_column, k)


def anonymize_l_diversity(df, feature_columns, sensitive_column, k, l):
    """
    A wrapper function passing l-diversity parameters to the anonymize function

    Parameters:
        df (pandas.DataFrame): The input DataFrame to be anonymized.
        feature_columns (list): List of column names to be used as quasi-identifiers for k-anonymity.
        sensitive_column (str): The column name containing sensitive information.
        k (int): The desired k-value for k-anonymity.
        l (int): The desired l-value for l-diversity.

    Returns:
        pandas.DataFrame: A new DataFrame with k-anonymized and l-diverse data.
    """
    return __anonymize(df, feature_columns, sensitive_column, k, l=l)


def anonymize_t_closeness(df, feature_columns, sensitive_column, k, t):
    """
    A wrapper function passing t-closeness parameters to the anonymize function

    Parameters:
        df (pandas.DataFrame): The input DataFrame to be anonymized.
        feature_columns (list): List of column names to be used as quasi-identifiers for k-anonymity.
        sensitive_column (str): The column name containing sensitive information.
        k (int): The desired k-value for k-anonymity.
        t (float): The desired t-closeness value.

    Returns:
        pandas.DataFrame: A new DataFrame with k-anonymized and t-close data.
    """
    return __anonymize(df, feature_columns, sensitive_column, k, t=t)


def aggregate(df, partitions, feature_columns, sensitive_column, max_partitions=None):
    """
    Aggregate quasi-identifier values within partitions.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        partitions (list): List of partitions to aggregate.
        feature_columns (list): List of column names representing the feature attributes.
        sensitive_column (str): The name of the column containing sensitive information.
        max_partitions (int, optional): Maximum number of partitions to process (default is None).

    Returns:
        pandas.DataFrame: Anonymized DataFrame with aggregated quasi-identifier values.
    """
    print("Changing quasi-identifiers values with the aggregation of their partition")
    newdf = df.copy()
    for partition in partitions:
        for column in feature_columns:
            if column != sensitive_column:
                newdf.loc[partition, column] = newdf.loc[partition, column].mean()
    showcase(newdf, 3, partitions)
    print("Anonymized dataset generated")
    return newdf