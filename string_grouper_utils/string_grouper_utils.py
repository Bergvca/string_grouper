import numpy as np
import pandas as pd
from typing import List, Optional, Union
from dateutil.parser import parse
from numbers import Number
from datetime import datetime
import re
from textwrap import indent


def new_group_rep_by_earliest_timestamp(grouped_data: pd.DataFrame,
                                        group_col: Union[str, int],
                                        record_id_col: Union[str, int],
                                        timestamps: Union[pd.Series, str, int],
                                        record_name_col: Optional[Union[str, int]] = None,
                                        **kwargs) -> Union[pd.DataFrame, pd.Series]:
    """
    Selects the oldest string in each group as group-representative.
    :param grouped_data: The grouped DataFrame
    :param group_col: The name or positional index of the column in grouped_data containing the groups
    :param record_id_col: The name or positional index of the column in grouped_data with all groups' members' IDs
    (This will appear in the output)
    :param timestamps: pandas.Series or the column name (str) or column positional index (int) in grouped_data
    This contains the timestamps of the strings to be grouped.
    :param record_name_col: (Optional) The name or positional index of the column in grouped_data with
    all groups' members' names. (This will appear in the output.)
    :param **kwargs: These are the same keyword arguments you would pass to dateutil.parser.parse.
    These help in interpreting the string input which is to be parsed into datetime datatypes, e.g. day first.

    FYI, the dateutil.parser.parse keyword arguments documentation follows:
    """
    if isinstance(timestamps, pd.Series):
        if len(grouped_data) != len(timestamps):
            raise Exception('Both grouped_data and timestamps must be pandas.Series of the same length.')
    else:
        timestamps = get_column(timestamps, grouped_data)
    weights = parse_timestamps(timestamps, **kwargs)
    return group_rep_transform('idxmin', weights, grouped_data, group_col, record_id_col, record_name_col)


def new_group_rep_by_completeness(grouped_data: pd.DataFrame,
                                  group_col: Union[str, int],
                                  record_id_col: Union[str, int],
                                  record_name_col: Optional[Union[str, int]] = None,
                                  tested_cols: Optional[Union[pd.DataFrame, List[Union[str, int]]]] = None
                                  ) -> Union[pd.DataFrame, pd.Series]:
    """
    Selects the string in the group with the most filled-in row/record as group-representative.
    :param grouped_data: The grouped DataFrame
    :param group_col: The name or positional index of the column in grouped_data containing the groups
    :param record_id_col: The name or positional index of the column in grouped_data with all groups' members' IDs
    (This will appear in the output)
    :param record_name_col: (Optional) The name or positional index of the column in grouped_data with
    all groups' members' names. (This will appear in the output.)
    :param tested_cols: (Optional) pandas.DataFrame or list of column names/indices of grouped_data whose
    filled-in statuses are used to determine the new group-representative.
    If it is None then the entire group_data itself is used
    The input DataFrame of fields of the strings to be grouped.
    """
    if isinstance(tested_cols, pd.DataFrame):
        if len(grouped_data) != len(tested_cols):
            raise Exception('Both grouped_data and tested_cols must be pandas.DataFrame of the same length.')
    elif tested_cols is not None:
        tested_cols = get_column(tested_cols, grouped_data)
    else:
        tested_cols = grouped_data

    def is_notnull_and_not_empty(x):
        if x == '' or pd.isnull(x):
            return 0
        else:
            return 1

    weights = tested_cols.applymap(is_notnull_and_not_empty).sum(axis=1)
    return group_rep_transform('idxmax', weights, grouped_data, group_col, record_id_col, record_name_col)


def new_group_rep_by_highest_weight(grouped_data: pd.DataFrame,
                                    group_col: Union[str, int],
                                    record_id_col: Union[str, int],
                                    weights: Union[pd.Series, str, int],
                                    record_name_col: Optional[Union[str, int]] = None,
                                    ) -> Union[pd.DataFrame, pd.Series]:
    """
    Selects the string in the group with the largest weight as group-representative.
    :param grouped_data: The grouped DataFrame
    :param group_col: The name or positional index of the column in grouped_data containing the groups
    :param record_id_col: The name or positional index of the column in grouped_data with all groups' members' IDs
    (This will appear in the output)
    :param weights: pandas.Series or the column name (str) or column positional index (int) in grouped_data
    containing the user-defined weights of the strings to be grouped
    :param record_name_col: (Optional) The name or positional index of the column in grouped_data with
    all groups' members' names. (This will appear in the output.)
    """
    if isinstance(weights, pd.Series):
        if len(grouped_data) != len(weights):
            raise Exception('Both grouped_data and weights must be pandas.Series of the same length.')
    else:
        weights = get_column(weights, grouped_data)
    return group_rep_transform('idxmax', weights, grouped_data, group_col, record_id_col, record_name_col)


def group_rep_transform(method: str,
                        weights: pd.Series,
                        grouped_data,
                        group_col,
                        record_id_col,
                        record_name_col) -> Union[pd.Series, pd.DataFrame]:
    group_of_master_id = get_column(group_col, grouped_data)
    group_of_master_id = group_of_master_id.rename('raw_group_id').reset_index().rename(columns={'index': 'weight'})
    group_of_master_id['weight'] = weights
    group_of_master_id['group_rep'] = \
        group_of_master_id.groupby('raw_group_id', sort=False)['weight'].transform(method)
    record_id_col = get_column(record_id_col, grouped_data)
    new_rep = record_id_col.iloc[group_of_master_id.group_rep].reset_index(drop=True).rename(None)
    if record_name_col is None:
        return new_rep
    else:
        record_name_col = get_column(record_name_col, grouped_data)
        new_rep_name = record_name_col.iloc[group_of_master_id.group_rep].reset_index(drop=True).rename(None)
        return pd.concat([new_rep, new_rep_name], axis=1)


def get_column(col: Union[str, int, List[Union[str, int]]], data: pd.DataFrame):
    if isinstance(col, str):
        return data.loc[:, col]
    elif isinstance(col, int):
        return data.iloc[:, col]
    elif isinstance(col, List):
        return pd.concat([get_column(m, data) for m in col], axis=1)


def parse_timestamps(timestamps: pd.Series, **kwargs) -> pd.Series:
    error_msg = f"timestamps must be a Series of date-like or datetime-like strings"
    error_msg += f" or datetime datatype or pandas Timestamp datatype or numbers"
    if is_series_of_type(str, timestamps):
        # if any of the strings is not datetime-like raise an exception
        if timestamps.to_frame().applymap(is_not_date).squeeze().any():
            raise Exception(error_msg)
        else:
            # convert strings to numpy datetime64
            return timestamps.copy().transform(lambda x: np.datetime64(parse(x, **kwargs)))
    elif is_series_of_type(type(pd.Timestamp('15-1-2000')), timestamps):
        # convert pandas Timestamps to numpy datetime64
        return timestamps.copy().transform(lambda x: x.to_numpy())
    elif is_series_of_type(datetime, timestamps):
        # convert python datetimes to numpy datetime64
        return timestamps.copy().transform(lambda x: np.datetime64(x))
    elif not is_series_of_type(Number, timestamps):
        raise Exception(error_msg)
    return timestamps


def is_not_date(string, fuzzy=True):
    """
    Return whether the string can be interpreted as a date.
    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return False
    except ValueError:
        return True


def is_series_of_type(what: type, series_to_test: pd.Series) -> bool:
    if series_to_test.to_frame().applymap(
                lambda x: not isinstance(x, what)
            ).squeeze().any():
        return False
    return True


# The following lines append the kwargs portion of the docstring of dateutil.parser.parse to
# the docstring of new_group_rep_by_earliest_timestamp:
parse_docstring_kwargs = re.search('The ``\*\*kwargs``.*?:return:', parse.__doc__, flags=re.DOTALL)
new_group_rep_by_earliest_timestamp.__doc__ = new_group_rep_by_earliest_timestamp.__doc__ + \
    indent(parse_docstring_kwargs.group(0)[:-9], '    ', lambda line: True)
