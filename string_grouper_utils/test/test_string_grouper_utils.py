import unittest
import pandas as pd
from dateutil.parser import parse
from string_grouper_utils.string_grouper_utils import new_group_rep_by_earliest_timestamp, \
    new_group_rep_by_completeness, new_group_rep_by_highest_weight


class SimpleExample(object):
    def __init__(self):
        self.customers_df = pd.DataFrame(
            [
                ('BB016741P', 'Mega Enterprises Corporation', 'Address0', 'Tel0', 'Description0', 0.2,
                    '2014-12-30 10:55:00-02:00', 'EE059082Q', 'Mega Enterprises Corp.'),
                ('CC082744L', 'Hyper Startup Incorporated', '', 'Tel1', '', 0.5, '2017-01-01 20:23:15-05:00',
                 'BB099931J', 'Hyper-Startup Inc.'),
                ('AA098762D', 'Hyper Startup Inc.', 'Address2', 'Tel2', 'Description2', 0.3,
                    '2020-10-20 15:29:30+02:00', 'BB099931J', 'Hyper-Startup Inc.'),
                ('BB099931J', 'Hyper-Startup Inc.', 'Address3', 'Tel3', 'Description3', 0.1,
                    '2013-07-01 03:34:45-05:00', 'BB099931J', 'Hyper-Startup Inc.'),
                ('HH072982K', 'Hyper Hyper Inc.', 'Address4', '', 'Description4', 0.9, '2005-09-11 11:56:00-07:00',
                    'HH072982K', 'Hyper Hyper Inc.'),
                ('EE059082Q', 'Mega Enterprises Corp.', 'Address5', 'Tel5', 'Description5', 1.0,
                    '1998-04-14 09:21:11+00:00', 'EE059082Q', 'Mega Enterprises Corp.')
            ],
            columns=('Customer ID', 'Customer Name', 'Address', 'Tel', 'Description', 'weight', 'timestamp',
                     'group ID', 'group name')
        )
        # new_group_rep_by_earliest_timestamp(customers_df, 'group ID', 'Customer ID', 'timestamp')
        self.expected_result_TS = pd.Series(
            [
                'EE059082Q',
                'BB099931J',
                'BB099931J',
                'BB099931J',
                'HH072982K',
                'EE059082Q',
            ]
        )
        # new_group_rep_by_earliest_timestamp(customers_df, 'group ID', 'Customer ID', 'timestamp', 'Customer Name')
        self.expected_result_T = pd.DataFrame(
            [
                ('EE059082Q', 'Mega Enterprises Corp.'),
                ('BB099931J', 'Hyper-Startup Inc.'),
                ('BB099931J', 'Hyper-Startup Inc.'),
                ('BB099931J', 'Hyper-Startup Inc.'),
                ('HH072982K', 'Hyper Hyper Inc.'),
                ('EE059082Q', 'Mega Enterprises Corp.')
            ]
        )
        # new_group_rep_by_earliest_timestamp(customers_df, 'group ID', 'Customer ID', 'weight', 'Customer Name')
        self.expected_result_TW = pd.DataFrame(
            [
                ('BB016741P', 'Mega Enterprises Corporation'),
                ('BB099931J', 'Hyper-Startup Inc.'),
                ('BB099931J', 'Hyper-Startup Inc.'),
                ('BB099931J', 'Hyper-Startup Inc.'),
                ('HH072982K', 'Hyper Hyper Inc.'),
                ('BB016741P', 'Mega Enterprises Corporation')
            ]
        )
        # new_group_rep_by_highest_weight(customers_df, 'group ID', 'Customer ID', 'weight', 'Customer Name')
        self.expected_result_W = pd.DataFrame(
            [
                ('EE059082Q', 'Mega Enterprises Corp.'),
                ('CC082744L', 'Hyper Startup Incorporated'),
                ('CC082744L', 'Hyper Startup Incorporated'),
                ('CC082744L', 'Hyper Startup Incorporated'),
                ('HH072982K', 'Hyper Hyper Inc.'),
                ('EE059082Q', 'Mega Enterprises Corp.')
            ]
        )
        # new_group_rep_by_highest_weight(customers_df, 'group ID', 'Customer ID', 'weight', 'Customer Name')
        self.expected_result_C = pd.DataFrame(
            [
                ('BB016741P', 'Mega Enterprises Corporation'),
                ('AA098762D', 'Hyper Startup Inc.'),
                ('AA098762D', 'Hyper Startup Inc.'),
                ('AA098762D', 'Hyper Startup Inc.'),
                ('HH072982K', 'Hyper Hyper Inc.'),
                ('BB016741P', 'Mega Enterprises Corporation')
            ]
        )


class StringGrouperUtilTest(unittest.TestCase):
    def test_group_rep_by_timestamp_return_series(self):
        """Should return a pd.Series object with the same length as the grouped_data. The series object will contain
        a list of groups whose group-representatives have the earliest timestamp of the group"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        pd.testing.assert_series_equal(
            simple_example.expected_result_TS,
            new_group_rep_by_earliest_timestamp(
                customers_df,
                'group ID',
                'Customer ID',
                'timestamp'
            )
        )

    def test_group_rep_by_timestamp_return_dataframe(self):
        """Should return a pd.DataFrame object with the same length as the grouped_data. The DataFrame object will contain
        a list of groups whose group-representatives have the earliest timestamp of the group"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        pd.testing.assert_frame_equal(
            simple_example.expected_result_T,
            new_group_rep_by_earliest_timestamp(
                customers_df,
                'group ID',
                'Customer ID',
                'timestamp',
                'Customer Name'
            )
        )

    def test_group_rep_by_timestamp_series_input(self):
        """Should return a pd.DataFrame object with the same length as the grouped_data. The DataFrame object will contain
        a list of groups whose group-representatives have the earliest timestamp of the group"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        pd.testing.assert_frame_equal(
            simple_example.expected_result_T,
            new_group_rep_by_earliest_timestamp(
                customers_df,
                'group ID',
                'Customer ID',
                customers_df['timestamp'],
                'Customer Name'
            )
        )

    def test_group_rep_by_timestamp_input_series_length(self):
        """Should raise an exception when timestamps series length is not the same as the length of grouped_data"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        with self.assertRaises(Exception):
            _ = new_group_rep_by_earliest_timestamp(
                customers_df,
                'group ID',
                'Customer ID',
                customers_df['timestamp'].iloc[:-2],
                'Customer Name'
            )

    def test_group_rep_by_timestamp_bad_input_timestamp_strings(self):
        """Should raise an exception when timestamps series of strings is not datetime-like"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        with self.assertRaises(Exception):
            _ = new_group_rep_by_earliest_timestamp(
                customers_df,
                'group ID',
                'Customer ID',
                customers_df['Customer ID'],
                'Customer Name'
            )

    def test_group_rep_by_timestamp_pandas_timestamps(self):
        """Should return a pd.DataFrame object with the same length as the grouped_data. The DataFrame object will contain
        a list of groups whose group-representatives have the earliest timestamp of the group"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        customers_df2 = customers_df.copy()
        customers_df2['timestamp'] = customers_df2['timestamp'].transform(lambda t: pd.Timestamp(t))
        pd.testing.assert_frame_equal(
            simple_example.expected_result_T,
            new_group_rep_by_earliest_timestamp(
                customers_df2,
                'group ID',
                'Customer ID',
                customers_df2['timestamp'],
                'Customer Name'
            )
        )

    def test_group_rep_by_timestamp_dateutil_timestamps(self):
        """Should return a pd.DataFrame object with the same length as the grouped_data. The DataFrame object will contain
        a list of groups whose group-representatives have the earliest timestamp of the group"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        customers_df2 = customers_df.copy()
        customers_df2['timestamp'] = customers_df2['timestamp'].transform(lambda t: parse(t))
        pd.testing.assert_frame_equal(
            simple_example.expected_result_T,
            new_group_rep_by_earliest_timestamp(
                customers_df2,
                'group ID',
                'Customer ID',
                customers_df2['timestamp'],
                'Customer Name'
            )
        )

    def test_group_rep_by_timestamp_bad_nonstring_timestamps(self):
        """Should raise an exception when not all provided timestamps are datetime-like or number-like"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        customers_df2 = customers_df.copy()
        customers_df2.at[0, 'timestamp'] = 1.0
        with self.assertRaises(Exception):
            _ = new_group_rep_by_earliest_timestamp(
                customers_df2,
                'group ID',
                'Customer ID',
                customers_df2['timestamp'],
                'Customer Name'
            )

    def test_group_rep_by_timestamp_input_numbers(self):
        """Should return a pd.DataFrame object with the same length as the grouped_data. The DataFrame object will contain
        a list of groups whose group-representatives have the earliest timestamp of the group"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        pd.testing.assert_frame_equal(
            simple_example.expected_result_TW,
            new_group_rep_by_earliest_timestamp(
                customers_df,
                'group ID',
                'Customer ID',
                customers_df['weight'],
                'Customer Name'
            )
        )

    def test_group_rep_by_weight(self):
        """Should return a pd.DataFrame object with the same length as the grouped_data. The DataFrame object will contain
        a list of groups whose group-representatives have the highest weight of the group"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        pd.testing.assert_frame_equal(
            simple_example.expected_result_W,
            new_group_rep_by_highest_weight(
                customers_df,
                'group ID',
                'Customer ID',
                'weight',
                'Customer Name'
            )
        )

    def test_group_rep_by_weight_input_series(self):
        """Should return a pd.DataFrame object with the same length as the grouped_data. The DataFrame object will contain
        a list of groups whose group-representatives have the highest weight of the group"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        pd.testing.assert_frame_equal(
            simple_example.expected_result_W,
            new_group_rep_by_highest_weight(
                customers_df,
                'group ID',
                'Customer ID',
                customers_df['weight'],
                'Customer Name'
            )
        )

    def test_group_rep_by_weight_input_series_length(self):
        """Should raise an exception when weights series length is not the same as the length of grouped_data"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        with self.assertRaises(Exception):
            _ = new_group_rep_by_highest_weight(
                customers_df,
                'group ID',
                'Customer ID',
                customers_df['weight'].iloc[:-2],
                'Customer Name'
            )

    def test_group_rep_by_completeness_column_list(self):
        """Should return a pd.DataFrame object with the same length as the grouped_data. The DataFrame object will contain
        a list of groups whose group-representatives have the most filled-in records of the group"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        pd.testing.assert_frame_equal(
            simple_example.expected_result_C,
            new_group_rep_by_completeness(
                customers_df,
                'group ID',
                'Customer ID',
                'Customer Name',
                [1, 2, 3, 4]
            )
        )

    def test_group_rep_by_completeness_no_columns(self):
        """Should return a pd.DataFrame object with the same length as the grouped_data. The DataFrame object will contain
        a list of groups whose group-representatives have the most filled-in records of the group"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        pd.testing.assert_frame_equal(
            simple_example.expected_result_C,
            new_group_rep_by_completeness(
                customers_df,
                'group ID',
                'Customer ID',
                'Customer Name'
            )
        )

    def test_group_rep_by_completeness_input_dataframe(self):
        """Should return a pd.DataFrame object with the same length as the grouped_data. The DataFrame object will contain
        a list of groups whose group-representatives have the most filled-in records of the group"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        pd.testing.assert_frame_equal(
            simple_example.expected_result_C,
            new_group_rep_by_completeness(
                customers_df,
                'group ID',
                'Customer ID',
                'Customer Name',
                customers_df
            )
        )

    def test_group_rep_by_completeness_input_dataframe_length(self):
        """Should raise an exception when tested_cols length is not the same as the length of grouped_data"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        with self.assertRaises(Exception):
            _ = new_group_rep_by_completeness(
                customers_df,
                'group ID',
                'Customer ID',
                'Customer Name',
                customers_df.iloc[:-2, :]
            )


if __name__ == '__main__':
    unittest.main()
