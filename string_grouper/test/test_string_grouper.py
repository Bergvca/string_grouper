import unittest
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from string_grouper.string_grouper import DEFAULT_MIN_SIMILARITY, \
    DEFAULT_REGEX, DEFAULT_NGRAM_SIZE, DEFAULT_N_PROCESSES, DEFAULT_IGNORE_CASE, \
    StringGrouperConfig, StringGrouper, StringGrouperNotFitException, \
    match_most_similar, group_similar_strings, match_strings, \
    compute_pairwise_similarities, DEFAULT_USE_SP_MATMUL_RS
from unittest.mock import patch, Mock


def mock_symmetrize_matrix(x: csr_matrix) -> csr_matrix:
    return x


def fix_row_order(df):
    """Sorts match rows canonically so DataFrames can be compared independently of row order"""
    return df.sort_values(['right_index', 'left_index']).reset_index(drop=True)


class SimpleExample(object):
    def __init__(self):
        self.customers_df = pd.DataFrame(
           [
              ('BB016741P', 'Mega Enterprises Corporation', 'Address0', 'Tel0', 'Description0', 0.2),
              ('CC082744L', 'Hyper Startup Incorporated', '', 'Tel1', '', 0.5),
              ('AA098762D', 'Hyper Startup Inc.', 'Address2', 'Tel2', 'Description2', 0.3),
              ('BB099931J', 'Hyper-Startup Inc.', 'Address3', 'Tel3', 'Description3', 0.1),
              ('HH072982K', 'Hyper Hyper Inc.', 'Address4', '', 'Description4', 0.9),
              ('EE059082Q', 'Mega Enterprises Corp.', 'Address5', 'Tel5', 'Description5', 1.0)
           ],
           columns=('Customer ID', 'Customer Name', 'Address', 'Tel', 'Description', 'weight')
        )
        self.customers_df2 = pd.DataFrame(
           [
              ('BB016741P', 'Mega Enterprises Corporation', 'Address0', 'Tel0', 'Description0', 0.2),
              ('CC082744L', 'Hyper Startup Incorporated', '', 'Tel1', '', 0.5),
              ('AA098762D', 'Hyper Startup Inc.', 'Address2', 'Tel2', 'Description2', 0.3),
              ('BB099931J', 'Hyper-Startup Inc.', 'Address3', 'Tel3', 'Description3', 0.1),
              ('DD012339M', 'HyperStartup Inc.', 'Address4', 'Tel4', 'Description4', 0.1),
              ('HH072982K', 'Hyper Hyper Inc.', 'Address5', '', 'Description5', 0.9),
              ('EE059082Q', 'Mega Enterprises Corp.', 'Address6', 'Tel6', 'Description6', 1.0)
           ],
           columns=('Customer ID', 'Customer Name', 'Address', 'Tel', 'Description', 'weight')
        )
        self.a_few_strings = pd.Series(['BB016741P', 'BB082744L', 'BB098762D', 'BB099931J', 'BB072982K', 'BB059082Q'])
        self.one_string = pd.Series(['BB0'])
        self.two_strings = pd.Series(['Hyper', 'Hyp'])
        self.whatever_series_1 = pd.Series(['whatever'])
        self.expected_result_with_zeroes = pd.DataFrame(
            [
                (1, 'Hyper Startup Incorporated', 0.08170638, 'whatever', 0),
                (0, 'Mega Enterprises Corporation', 0., 'whatever', 0),
                (2, 'Hyper Startup Inc.', 0., 'whatever', 0),
                (3, 'Hyper-Startup Inc.', 0., 'whatever', 0),
                (4, 'Hyper Hyper Inc.', 0., 'whatever', 0),
                (5, 'Mega Enterprises Corp.', 0., 'whatever', 0)
            ],
            columns=['left_index', 'left_Customer Name', 'similarity', 'right_side', 'right_index']
        )
        self.expected_result_centroid = pd.Series(
            [
                'Mega Enterprises Corporation',
                'Hyper Startup Inc.',
                'Hyper Startup Inc.',
                'Hyper Startup Inc.',
                'Hyper Hyper Inc.',
                'Mega Enterprises Corporation'
            ],
            name='group_rep_Customer Name'
        )
        self.expected_result_centroid_with_index_col = pd.DataFrame(
            [
                (0, 'Mega Enterprises Corporation'),
                (2, 'Hyper Startup Inc.'),
                (2, 'Hyper Startup Inc.'),
                (2, 'Hyper Startup Inc.'),
                (4, 'Hyper Hyper Inc.'),
                (0, 'Mega Enterprises Corporation')
            ],
            columns=['group_rep_index', 'group_rep_Customer Name']
        )
        self.expected_result_first = pd.Series(
            [
                 'Mega Enterprises Corporation',
                 'Hyper Startup Incorporated',
                 'Hyper Startup Incorporated',
                 'Hyper Startup Incorporated',
                 'Hyper Hyper Inc.',
                 'Mega Enterprises Corporation'
            ],
            name='group_rep_Customer Name'
        )


class StringGrouperConfigTest(unittest.TestCase):

    def test_config_defaults(self):
        """Empty initialisation should set default values"""
        config = StringGrouperConfig()
        self.assertEqual(config.min_similarity, DEFAULT_MIN_SIMILARITY)
        self.assertEqual(config.max_n_matches, 20)
        self.assertEqual(config.regex, DEFAULT_REGEX)
        self.assertEqual(config.ngram_size, DEFAULT_NGRAM_SIZE)
        self.assertEqual(config.number_of_processes, DEFAULT_N_PROCESSES)
        self.assertEqual(config.ignore_case, DEFAULT_IGNORE_CASE)
        self.assertEqual(config.use_sp_matmul_rs, DEFAULT_USE_SP_MATMUL_RS)

    def test_config_immutable(self):
        """Configurations should be immutable"""
        config = StringGrouperConfig()
        with self.assertRaises(Exception) as _:
            config.min_similarity = 0.1

    def test_config_non_default_values(self):
        """Configurations should be immutable"""
        config = StringGrouperConfig(min_similarity=0.1, max_n_matches=100, number_of_processes=1)
        self.assertEqual(0.1, config.min_similarity)
        self.assertEqual(100, config.max_n_matches)
        self.assertEqual(1, config.number_of_processes)


class StringGrouperTest(unittest.TestCase):

    def test_auto_blocking_single_DataFrame(self):
        """tests whether automatic blocking yields consistent results"""
        # This function will force an OverflowError to occur when
        # the input Series have a combined length above a given number:
        # OverflowThreshold.  This will in turn trigger automatic splitting
        # of the Series/matrices into smaller blocks when n_blocks = None

        simple_example = SimpleExample()
        df1 = simple_example.customers_df2['Customer Name']

        # first do manual blocking
        sg = StringGrouper(df1, min_similarity=0.1, use_sp_matmul_rs=False)
        pd.testing.assert_series_equal(sg.master, df1)
        self.assertEqual(sg.duplicates, None)

        matches = fix_row_order(sg.match_strings(df1, n_blocks=(1, 1), use_sp_matmul_rs=False))
        self.assertEqual(sg._config.n_blocks, (1, 1))

        # Create a custom wrapper for this StringGrouper instance's
        # _build_matches() method which will later be used to
        # mock _build_matches().
        # Note that we have  to  define  the  wrapper  here  because
        # _build_matches() is a non-static function of StringGrouper
        # and needs access to the specific StringGrouper instance sg
        # created here.
        def mock_build_matches(OverflowThreshold,
                               real_build_matches=sg._build_matches):
            def wrapper(left_matrix,
                        right_matrix,
                        nnz_rows=None,
                        sort=True):
                if (left_matrix.shape[0] + right_matrix.shape[0]) > \
                        OverflowThreshold:
                    raise OverflowError
                return real_build_matches(left_matrix, right_matrix, None)
            return wrapper

        def do_test_with(OverflowThreshold):
            nonlocal sg  # allows reference to sg, as sg will be modified below
            # Now let us mock sg._build_matches:
            sg._build_matches = Mock(side_effect=mock_build_matches(OverflowThreshold))
            sg.clear_data()
            matches_auto = fix_row_order(sg.match_strings(df1, n_blocks=None))
            pd.testing.assert_series_equal(sg.master, df1)
            pd.testing.assert_frame_equal(matches, matches_auto)
            self.assertEqual(sg._config.n_blocks, None)
            # Note that _build_matches is called more than once if and only if
            # a split occurred (that is, there was more than one pair of
            # matrix-blocks multiplied)
            if len(sg._left_Series) + len(sg._right_Series) > \
                    OverflowThreshold:
                # Assert that split occurred:
                self.assertGreater(sg._build_matches.call_count, 1)
            else:
                # Assert that split did not occur:
                self.assertEqual(sg._build_matches.call_count, 1)

        # now test auto blocking by forcing an OverflowError when the
        # combined Series' lengths is greater than 10, 5, 3, 2

        do_test_with(OverflowThreshold=100)  # does not trigger auto blocking
        do_test_with(OverflowThreshold=30)
        do_test_with(OverflowThreshold=20)
        do_test_with(OverflowThreshold=15)
        # do_test_with(OverflowThreshold=12)

    def test_n_blocks_single_DataFrame(self):
        """tests whether manual blocking yields consistent results"""
        simple_example = SimpleExample()
        df1 = simple_example.customers_df2['Customer Name']

        matches11 = fix_row_order(match_strings(df1, min_similarity=0.1))

        matches12 = fix_row_order(
            match_strings(df1, n_blocks=(1, 2), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches12)

        matches13 = fix_row_order(
            match_strings(df1, n_blocks=(1, 3), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches13)

        matches14 = fix_row_order(
            match_strings(df1, n_blocks=(1, 4), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches14)

        matches15 = fix_row_order(
            match_strings(df1, n_blocks=(1, 5), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches15)

        matches16 = fix_row_order(
            match_strings(df1, n_blocks=(1, 6), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches16)

        matches17 = fix_row_order(
            match_strings(df1, n_blocks=(1, 7), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches17)

        matches18 = fix_row_order(
            match_strings(df1, n_blocks=(1, 8), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches18)

        matches21 = fix_row_order(
            match_strings(df1, n_blocks=(2, 1), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches21)

        matches22 = fix_row_order(
            match_strings(df1, n_blocks=(2, 2), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches22)

        matches32 = fix_row_order(
            match_strings(df1, n_blocks=(3, 2), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches32)

        # Create a custom wrapper for this StringGrouper instance's
        # _build_matches() method which will later be used to
        # mock _build_matches().
        # Note that we have  to  define  the  wrapper  here  because
        # _build_matches() is a non-static function of StringGrouper
        # and needs access to the specific StringGrouper instance sg
        # created here.
        sg = StringGrouper(df1, min_similarity=0.1)

        def mock_build_matches(OverflowThreshold,
                               real_build_matches=sg._build_matches):
            def wrapper(left_matrix,
                        right_matrix,
                        nnz_rows=None,
                        sort=True):
                if (left_matrix.shape[0] + right_matrix.shape[0]) > \
                        OverflowThreshold:
                    raise OverflowError
                return real_build_matches(left_matrix, right_matrix, None)
            return wrapper

        def test_overflow_error_with(OverflowThreshold, n_blocks):
            nonlocal sg
            sg._build_matches = Mock(side_effect=mock_build_matches(OverflowThreshold))
            sg.clear_data()
            max_left_block_size = (len(df1)//n_blocks[0]
                                   + (1 if len(df1) % n_blocks[0] > 0 else 0))
            max_right_block_size = (len(df1)//n_blocks[1]
                                    + (1 if len(df1) % n_blocks[1] > 0 else 0))
            if (max_left_block_size + max_right_block_size) > OverflowThreshold:
                with self.assertRaises(Exception):
                    _ = sg.match_strings(df1, n_blocks=n_blocks, use_sp_matmul_rs=False)
            else:
                matches_manual = fix_row_order(sg.match_strings(df1, n_blocks=n_blocks, use_sp_matmul_rs=False))
                pd.testing.assert_frame_equal(matches11, matches_manual)

        test_overflow_error_with(OverflowThreshold=20, n_blocks=(1, 1))
        test_overflow_error_with(OverflowThreshold=20, n_blocks=(1, 1))
        test_overflow_error_with(OverflowThreshold=20, n_blocks=(2, 1))
        test_overflow_error_with(OverflowThreshold=20, n_blocks=(1, 2))
        test_overflow_error_with(OverflowThreshold=20, n_blocks=(4, 4))

    def test_n_blocks_both_DataFrames(self):
        """tests whether manual blocking yields consistent results"""
        simple_example = SimpleExample()
        df1 = simple_example.customers_df['Customer Name']
        df2 = simple_example.customers_df2['Customer Name']

        matches11 = fix_row_order(match_strings(df1, df2, min_similarity=0.1))

        matches12 = fix_row_order(
            match_strings(df1, df2, n_blocks=(1, 2), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches12)

        matches13 = fix_row_order(
            match_strings(df1, df2, n_blocks=(1, 3), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches13)

        matches14 = fix_row_order(
            match_strings(df1, df2, n_blocks=(1, 4), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches14)

        matches15 = fix_row_order(
            match_strings(df1, df2, n_blocks=(1, 5), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches15)

        matches16 = fix_row_order(
            match_strings(df1, df2, n_blocks=(1, 6), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches16)

        matches17 = fix_row_order(
            match_strings(df1, df2, n_blocks=(1, 7), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches17)

        matches18 = fix_row_order(
            match_strings(df1, df2, n_blocks=(1, 8), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches18)

        matches21 = fix_row_order(
            match_strings(df1, df2, n_blocks=(2, 1), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches21)

        matches22 = fix_row_order(
            match_strings(df1, df2, n_blocks=(2, 2), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches22)

        matches32 = fix_row_order(
            match_strings(df1, df2, n_blocks=(3, 2), min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches11, matches32)

    def test_n_blocks_bad_option_value(self):
        """Tests that bad option values for n_blocks are caught, regardless of the backend"""
        simple_example = SimpleExample()
        df1 = simple_example.customers_df2['Customer Name']
        for use_rs in (True, False):
            for bad_n_blocks in (2, (0, 2), (1, 2.5), (1, 2, 3), (1, )):
                with self.assertRaises(Exception) as cm:
                    _ = match_strings(df1, n_blocks=bad_n_blocks, use_sp_matmul_rs=use_rs)
                self.assertIn('tuple of 2 integers greater than 0', str(cm.exception))

    def test_tfidf_dtype_bad_option_value(self):
        """Tests that bad option values for n_blocks are caught"""
        simple_example = SimpleExample()
        df1 = simple_example.customers_df2['Customer Name']
        with self.assertRaises(Exception):
            _ = match_strings(df1, tfidf_matrix_dtype=None)
        with self.assertRaises(Exception):
            _ = match_strings(df1, tfidf_matrix_dtype=0)
        with self.assertRaises(Exception):
            _ = match_strings(df1, tfidf_matrix_dtype='whatever')

    def test_compute_pairwise_similarities(self):
        """tests the high-level function compute_pairwise_similarities"""
        simple_example = SimpleExample()
        df1 = simple_example.customers_df['Customer Name']
        df2 = simple_example.expected_result_centroid
        similarities = compute_pairwise_similarities(df1, df2)
        expected_result = pd.Series(
            [
                1.0,
                0.6336195351561589,
                1.0000000000000004,
                1.0000000000000004,
                1.0,
                0.826462625999832
            ],
            name='similarity'
        )
        expected_result = expected_result.astype(np.float64)
        pd.testing.assert_series_equal(expected_result, similarities)
        sg = StringGrouper(df1, df2)
        similarities = sg.compute_pairwise_similarities(df1, df2)
        pd.testing.assert_series_equal(expected_result, similarities)

    def test_compute_pairwise_similarities_data_integrity(self):
        """tests that an exception is raised whenever the lengths of the two input series of the high-level function
        compute_pairwise_similarities are unequal"""
        simple_example = SimpleExample()
        df1 = simple_example.customers_df['Customer Name']
        df2 = simple_example.expected_result_centroid
        with self.assertRaises(Exception):
            _ = compute_pairwise_similarities(df1, df2[:-2])

    @patch('string_grouper.string_grouper.StringGrouper')
    def test_group_similar_strings(self, mock_StringGouper):
        """mocks StringGrouper to test if the high-level function group_similar_strings utilizes it as expected"""
        mock_StringGrouper_instance = mock_StringGouper.return_value
        mock_StringGrouper_instance.fit.return_value = mock_StringGrouper_instance
        mock_StringGrouper_instance.get_groups.return_value = 'whatever'

        test_series_1 = None
        test_series_id_1 = None
        df = group_similar_strings(
                test_series_1,
                string_ids=test_series_id_1
            )

        mock_StringGrouper_instance.fit.assert_called_once()
        mock_StringGrouper_instance.get_groups.assert_called_once()
        self.assertEqual(df, 'whatever')

    @patch('string_grouper.string_grouper.StringGrouper')
    def test_match_most_similar(self, mock_StringGouper):
        """mocks StringGrouper to test if the high-level function match_most_similar utilizes it as expected"""
        mock_StringGrouper_instance = mock_StringGouper.return_value
        mock_StringGrouper_instance.fit.return_value = mock_StringGrouper_instance
        mock_StringGrouper_instance.get_groups.return_value = 'whatever'

        test_series_1 = None
        test_series_2 = None
        test_series_id_1 = None
        test_series_id_2 = None
        df = match_most_similar(
                test_series_1,
                test_series_2,
                master_id=test_series_id_1,
                duplicates_id=test_series_id_2
            )

        mock_StringGrouper_instance.fit.assert_called_once()
        mock_StringGrouper_instance.get_groups.assert_called_once()
        self.assertEqual(df, 'whatever')

    @patch('string_grouper.string_grouper.StringGrouper')
    def test_match_strings(self, mock_StringGouper):
        """mocks StringGrouper to test if the high-level function match_strings utilizes it as expected"""
        mock_StringGrouper_instance = mock_StringGouper.return_value
        mock_StringGrouper_instance.fit.return_value = mock_StringGrouper_instance
        mock_StringGrouper_instance.get_matches.return_value = 'whatever'

        test_series_1 = None
        test_series_id_1 = None
        df = match_strings(test_series_1, master_id=test_series_id_1)

        mock_StringGrouper_instance.fit.assert_called_once()
        mock_StringGrouper_instance.get_matches.assert_called_once()
        self.assertEqual(df, 'whatever')

    @patch(
        'string_grouper.string_grouper.StringGrouper._fix_diagonal',
        side_effect=mock_symmetrize_matrix
    )
    def test_match_list_diagonal_without_the_fix(self, mock_fix_diagonal):
        """test fails whenever _matches_list's number of self-joins is not equal to the number of strings"""
        # This bug is difficult to reproduce -- I mostly encounter it while working with very large datasets;
        # for small datasets setting max_n_matches=1 reproduces the bug
        simple_example = SimpleExample()
        df = simple_example.customers_df['Customer Name']
        matches = match_strings(df, max_n_matches=1)
        mock_fix_diagonal.assert_called_once()
        num_self_joins = len(matches[matches['left_index'] == matches['right_index']])
        num_strings = len(df)
        self.assertNotEqual(num_self_joins, num_strings)

    def test_match_list_diagonal(self):
        """This test ensures that all self-joins are present"""
        # This bug is difficult to reproduce -- I mostly encounter it while working with very large datasets;
        # for small datasets setting max_n_matches=1 reproduces the bug
        simple_example = SimpleExample()
        df = simple_example.customers_df['Customer Name']
        matches = match_strings(df, max_n_matches=1)
        num_self_joins = len(matches[matches['left_index'] == matches['right_index']])
        num_strings = len(df)
        self.assertEqual(num_self_joins, num_strings)

    def test_zero_min_similarity(self):
        """Since sparse matrices exclude zero elements, this test ensures that zero similarity matches are
        returned when min_similarity <= 0.  A bug related to this was first pointed out by @nbcvijanovic"""
        simple_example = SimpleExample()
        s_master = simple_example.customers_df['Customer Name']
        s_dup = simple_example.whatever_series_1
        matches = match_strings(s_master, s_dup, min_similarity=0)
        pd.testing.assert_frame_equal(simple_example.expected_result_with_zeroes, matches)

    def test_get_non_matches_empty_case(self):
        """This test ensures that _get_non_matches() returns an empty DataFrame when all pairs of strings match"""
        simple_example = SimpleExample()
        s_master = simple_example.a_few_strings
        s_dup = simple_example.one_string
        sg = StringGrouper(s_master, s_dup, max_n_matches=len(s_master), min_similarity=0).fit()
        self.assertTrue(sg._get_non_matches_list().empty)

    def test_n_grams_case_unchanged(self):
        """Should return all ngrams in a string with case"""
        test_series = pd.Series(pd.Series(['aaa']))
        # Explicit do not ignore case
        sg = StringGrouper(test_series, ignore_case=False)
        expected_result = ['McD', 'cDo', 'Don', 'ona', 'nal', 'ald', 'lds']
        self.assertListEqual(expected_result, sg.n_grams('McDonalds'))

    def test_n_grams_ignore_case_to_lower(self):
        """Should return all case insensitive ngrams in a string"""
        test_series = pd.Series(pd.Series(['aaa']))
        # Explicit ignore case
        sg = StringGrouper(test_series, ignore_case=True)
        expected_result = ['mcd', 'cdo', 'don', 'ona', 'nal', 'ald', 'lds']
        self.assertListEqual(expected_result, sg.n_grams('McDonalds'))

    def test_n_grams_ignore_case_to_lower_with_defaults(self):
        """Should return all case insensitive ngrams in a string"""
        test_series = pd.Series(pd.Series(['aaa']))
        # Implicit default case (i.e. default behaviour)
        sg = StringGrouper(test_series)
        expected_result = ['mcd', 'cdo', 'don', 'ona', 'nal', 'ald', 'lds']
        self.assertListEqual(expected_result, sg.n_grams('McDonalds'))

    def test_build_matrix(self):
        """Should create a csr matrix only master"""
        test_series = pd.Series(['foo', 'bar', 'baz'])
        sg = StringGrouper(test_series)
        master, dupe = sg._get_tf_idf_matrices()
        c = csr_matrix([[0., 0., 1.],
                        [1., 0., 0.],
                        [0., 1., 0.]])
        np.testing.assert_array_equal(c.toarray(), master.toarray())
        np.testing.assert_array_equal(c.toarray(), dupe.toarray())

    def test_build_matrix_master_and_duplicates(self):
        """Should create a csr matrix for master and duplicates"""
        test_series_1 = pd.Series(['foo', 'bar', 'baz'])
        test_series_2 = pd.Series(['foo', 'bar', 'bop'])
        sg = StringGrouper(test_series_1, test_series_2)
        master, dupe = sg._get_tf_idf_matrices()
        master_expected = csr_matrix([[0., 0., 0., 1.],
                                     [1., 0., 0., 0.],
                                     [0., 1., 0., 0.]])
        dupes_expected = csr_matrix([[0., 0., 0., 1.],
                                     [1., 0., 0., 0.],
                                     [0., 0., 1., 0.]])

        np.testing.assert_array_equal(master_expected.toarray(), master.toarray())
        np.testing.assert_array_equal(dupes_expected.toarray(), dupe.toarray())

    def test_build_matches(self):
        """Should create the cosine similarity matrix of two series"""
        test_series_1 = pd.Series(['foo', 'bar', 'baz'])
        test_series_2 = pd.Series(['foo', 'bar', 'bop'])
        sg = StringGrouper(test_series_1, test_series_2)
        master, dupe = sg._get_tf_idf_matrices()

        expected_matches = np.array([[1., 0., 0.],
                                     [0., 1., 0.],
                                     [0., 0., 0.]])
        np.testing.assert_array_equal(expected_matches, sg._build_matches(master, dupe, None).toarray())

    def test_build_matches_list(self):
        """Should create the cosine similarity matrix of two series"""
        test_series_1 = pd.Series(['foo', 'bar', 'baz'])
        test_series_2 = pd.Series(['foo', 'bar', 'bop'])
        sg = StringGrouper(test_series_1, test_series_2)
        sg = sg.fit()
        master = [0, 1]
        dupe_side = [0, 1]
        similarity = [1.0, 1.0]
        expected_df = pd.DataFrame({'master_side': master, 'dupe_side': dupe_side, 'similarity': similarity})
        expected_df.loc[:, 'similarity'] = expected_df.loc[:, 'similarity'].astype(sg._config.tfidf_matrix_dtype)
        pd.testing.assert_frame_equal(expected_df, sg._matches_list)

    def test_case_insensitive_build_matches_list(self):
        """Should create the cosine similarity matrix of two case insensitive series"""
        test_series_1 = pd.Series(['foo', 'BAR', 'baz'])
        test_series_2 = pd.Series(['FOO', 'bar', 'bop'])
        sg = StringGrouper(test_series_1, test_series_2)
        sg = sg.fit()
        master = [0, 1]
        dupe_side = [0, 1]
        similarity = [1.0, 1.0]
        expected_df = pd.DataFrame({'master_side': master, 'dupe_side': dupe_side, 'similarity': similarity})
        expected_df.loc[:, 'similarity'] = expected_df.loc[:, 'similarity'].astype(sg._config.tfidf_matrix_dtype)
        pd.testing.assert_frame_equal(expected_df, sg._matches_list)

    def test_get_matches_two_dataframes(self):
        test_series_1 = pd.Series(['foo', 'bar', 'baz'])
        test_series_2 = pd.Series(['foo', 'bar', 'bop'])
        sg = StringGrouper(test_series_1, test_series_2).fit()
        left_side = ['foo', 'bar']
        left_index = [0, 1]
        right_side = ['foo', 'bar']
        right_index = [0, 1]
        similarity = [1.0, 1.0]
        expected_df = pd.DataFrame({'left_index': left_index, 'left_side': left_side,
                                    'similarity': similarity,
                                    'right_side': right_side, 'right_index': right_index})
        expected_df.loc[:, 'similarity'] = expected_df.loc[:, 'similarity'].astype(sg._config.tfidf_matrix_dtype)
        pd.testing.assert_frame_equal(expected_df, sg.get_matches())

    def test_get_matches_single(self):
        test_series_1 = pd.Series(['foo', 'bar', 'baz', 'foo'])
        sg = StringGrouper(test_series_1)
        sg = sg.fit()
        left_side = ['foo', 'foo', 'bar', 'baz', 'foo', 'foo']
        right_side = ['foo', 'foo', 'bar', 'baz', 'foo', 'foo']
        right_index = [0, 3, 1, 2, 0, 3]
        left_index = [0, 0, 1, 2, 3, 3]
        similarity = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        expected_df = pd.DataFrame({'left_index': left_index, 'left_side': left_side,
                                    'similarity': similarity,
                                    'right_side': right_side, 'right_index': right_index})
        expected_df.loc[:, 'similarity'] = expected_df.loc[:, 'similarity'].astype(sg._config.tfidf_matrix_dtype)
        pd.testing.assert_frame_equal(expected_df, sg.get_matches())

    def test_get_matches_1_series_1_id_series(self):
        test_series_1 = pd.Series(['foo', 'bar', 'baz', 'foo'])
        test_series_id_1 = pd.Series(['A0', 'A1', 'A2', 'A3'])
        sg = StringGrouper(test_series_1, master_id=test_series_id_1)
        sg = sg.fit()
        right_side = ['foo', 'foo', 'bar', 'baz', 'foo', 'foo']
        right_side_id = ['A0', 'A3', 'A1', 'A2', 'A0', 'A3']
        right_index = [0, 3, 1, 2, 0, 3]
        left_side = ['foo', 'foo', 'bar', 'baz', 'foo', 'foo']
        left_side_id = ['A0', 'A0', 'A1', 'A2', 'A3', 'A3']
        left_index = [0, 0, 1, 2, 3, 3]
        similarity = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        similarity = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        expected_df = pd.DataFrame({'left_index': left_index, 'left_side': left_side, 'left_id': left_side_id,
                                    'similarity': similarity,
                                    'right_id': right_side_id, 'right_side': right_side, 'right_index': right_index})
        expected_df.loc[:, 'similarity'] = expected_df.loc[:, 'similarity'].astype(sg._config.tfidf_matrix_dtype)
        pd.testing.assert_frame_equal(expected_df, sg.get_matches())

    def test_get_matches_2_series_2_id_series(self):
        test_series_1 = pd.Series(['foo', 'bar', 'baz'])
        test_series_id_1 = pd.Series(['A0', 'A1', 'A2'])
        test_series_2 = pd.Series(['foo', 'bar', 'bop'])
        test_series_id_2 = pd.Series(['B0', 'B1', 'B2'])
        sg = StringGrouper(test_series_1, test_series_2, duplicates_id=test_series_id_2,
                           master_id=test_series_id_1).fit()
        left_side = ['foo', 'bar']
        left_side_id = ['A0', 'A1']
        left_index = [0, 1]
        right_side = ['foo', 'bar']
        right_side_id = ['B0', 'B1']
        right_index = [0, 1]
        similarity = [1.0, 1.0]
        expected_df = pd.DataFrame({'left_index': left_index, 'left_side': left_side, 'left_id': left_side_id,
                                    'similarity': similarity,
                                    'right_id': right_side_id, 'right_side': right_side, 'right_index': right_index})
        expected_df.loc[:, 'similarity'] = expected_df.loc[:, 'similarity'].astype(sg._config.tfidf_matrix_dtype)
        pd.testing.assert_frame_equal(expected_df, sg.get_matches())

    def test_get_matches_raises_exception_if_unexpected_options_given(self):
        # When the input id data does not correspond with its string data:
        test_series_1 = pd.Series(['foo', 'bar', 'baz'])
        bad_test_series_id_1 = pd.Series(['A0', 'A1'])
        good_test_series_id_1 = pd.Series(['A0', 'A1', 'A2'])
        test_series_2 = pd.Series(['foo', 'bar', 'bop'])
        bad_test_series_id_2 = pd.Series(['B0', 'B1'])
        good_test_series_id_2 = pd.Series(['B0', 'B1', 'B2'])
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, master_id=bad_test_series_id_1)
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, duplicates=test_series_2, duplicates_id=bad_test_series_id_2,
                              master_id=good_test_series_id_1)

        # When the input data is ok but the option combinations are invalid:
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, test_series_2, master_id=good_test_series_id_1)
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, test_series_2, duplicates_id=good_test_series_id_2)
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, duplicates_id=good_test_series_id_2)
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, master_id=good_test_series_id_1, duplicates_id=good_test_series_id_2)
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, master_id=good_test_series_id_1, ignore_index=True, replace_na=True)
        # Here we force an exception by making the number of index-levels of duplicates different from master:
        # and setting replace_na=True
        test_series_2.index = pd.MultiIndex.from_tuples(list(zip(list('ABC'), [0, 1, 2])))
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, duplicates=test_series_2, replace_na=True)

    def test_get_groups_single_df_group_rep_default(self):
        """Should return a pd.Series object with the same length as the original df. The series object will contain
        a list of the grouped strings"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        pd.testing.assert_series_equal(
            simple_example.expected_result_centroid,
            group_similar_strings(
                customers_df['Customer Name'],
                min_similarity=0.6,
                ignore_index=True
            )
        )
        sg = StringGrouper(customers_df['Customer Name'])
        pd.testing.assert_series_equal(
            simple_example.expected_result_centroid,
            sg.group_similar_strings(
                customers_df['Customer Name'],
                min_similarity=0.6,
                ignore_index=True
            )
        )

    def test_get_groups_single_valued_series(self):
        """This test ensures that get_groups() returns a single-valued DataFrame or Series object
        since the input-series is also single-valued.  This test was created in response to a bug discovered
        by George Walker"""
        pd.testing.assert_frame_equal(
            pd.DataFrame([(0, "hello")], columns=['group_rep_index', 'group_rep']),
            group_similar_strings(
                pd.Series(["hello"]),
                min_similarity=0.6
            )
        )
        pd.testing.assert_series_equal(
            pd.Series(["hello"], name='group_rep'),
            group_similar_strings(
                pd.Series(["hello"]),
                min_similarity=0.6,
                ignore_index=True
            )
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame([(0, "hello")], columns=['most_similar_index', 'most_similar_master']),
            match_most_similar(
                pd.Series(["hello"]),
                pd.Series(["hello"]),
                min_similarity=0.6
            )
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame([(0, "hello")], columns=['most_similar_index', 'most_similar_master']),
            match_most_similar(
                pd.Series(["hello"]),
                pd.Series(["hello"]),
                min_similarity=0.6,
                max_n_matches=20
            )
        )
        pd.testing.assert_series_equal(
            pd.Series(["hello"], name='most_similar_master'),
            match_most_similar(
                pd.Series(["hello"]),
                pd.Series(["hello"]),
                min_similarity=0.6,
                ignore_index=True
            )
        )

    def test_get_groups_single_df_keep_index(self):
        """Should return a pd.Series object with the same length as the original df. The series object will contain
        a list of the grouped strings with their indexes displayed in columns"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        pd.testing.assert_frame_equal(
            simple_example.expected_result_centroid_with_index_col,
            group_similar_strings(
                customers_df['Customer Name'],
                min_similarity=0.6,
                ignore_index=False
            )
        )

    def test_get_groups_single_df_group_rep_centroid(self):
        """Should return a pd.Series object with the same length as the original df. The series object will contain
        a list of the grouped strings"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        pd.testing.assert_series_equal(
            simple_example.expected_result_first,
            group_similar_strings(
                customers_df['Customer Name'],
                group_rep='first',
                min_similarity=0.6,
                ignore_index=True
            )
        )

    def test_get_groups_single_df_group_rep_bad_option_value(self):
        """Should raise an exception when group_rep value given is neither 'centroid' nor 'first'"""
        simple_example = SimpleExample()
        customers_df = simple_example.customers_df
        with self.assertRaises(Exception):
            _ = group_similar_strings(
                    customers_df['Customer Name'],
                    group_rep='nonsense',
                    min_similarity=0.6
                )

    def test_get_groups_single_df(self):
        """Should return a pd.Series object with the same length as the original df. The series object will contain
        a list of the grouped strings"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        sg = StringGrouper(test_series_1, ignore_index=True)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.Series(['foooo', 'bar', 'baz', 'foooo'], name='group_rep')
        pd.testing.assert_series_equal(expected_result, result)

    def test_get_groups_1_string_series_1_id_series(self):
        """Should return a pd.DataFrame object with the same length as the original df. The series object will contain
        a list of the grouped strings"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        test_series_id_1 = pd.Series(['A0', 'A1', 'A2', 'A3'])
        sg = StringGrouper(test_series_1, master_id=test_series_id_1, ignore_index=True)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.DataFrame(list(zip(['A0', 'A1', 'A2', 'A0'], ['foooo', 'bar', 'baz', 'foooo'])),
                                       columns=['group_rep_id', 'group_rep'])
        pd.testing.assert_frame_equal(expected_result, result)

    def test_get_groups_two_df(self):
        """Should return a pd.Series object with the length of the dupes. The series will contain the master string
        that matches the dupe with the highest similarity"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        sg = StringGrouper(test_series_1, test_series_2, ignore_index=True)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.Series(['foooo', 'bar', 'baz', 'foooo'], name='most_similar_master')
        pd.testing.assert_series_equal(expected_result, result)
        result = sg.match_most_similar(test_series_1, test_series_2, max_n_matches=3)
        pd.testing.assert_series_equal(expected_result, result)

    def test_get_groups_2_string_series_2_id_series(self):
        """Should return a pd.DataFrame object with the length of the dupes. The series will contain the master string
        that matches the dupe with the highest similarity"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        test_series_id_1 = pd.Series(['A0', 'A1', 'A2'])
        test_series_id_2 = pd.Series(['B0', 'B1', 'B2', 'B3'])
        sg = StringGrouper(test_series_1,
                           test_series_2,
                           master_id=test_series_id_1,
                           duplicates_id=test_series_id_2,
                           ignore_index=True)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.DataFrame(list(zip(['A0', 'A1', 'A2', 'A0'], ['foooo', 'bar', 'baz', 'foooo'])),
                                       columns=['most_similar_master_id', 'most_similar_master'])
        pd.testing.assert_frame_equal(expected_result, result)

    def test_get_groups_2_string_series_2_numeric_id_series_with_missing_master_value(self):
        """Should return a pd.DataFrame object with the length of the dupes. The series will contain the master string
        that matches the dupe with the highest similarity"""
        test_series_1 = pd.Series(['foooo', 'bar', 'foooo'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        test_series_id_1 = pd.Series([0, 1, 2], dtype = "Int64")
        test_series_id_2 = pd.Series([100, 101, 102, 103], dtype = "Int64")
        sg = StringGrouper(test_series_1,
                           test_series_2,
                           master_id=test_series_id_1,
                           duplicates_id=test_series_id_2,
                           ignore_index=True)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.DataFrame(list(zip([0, 1, 102, 0], ['foooo', 'bar', 'baz', 'foooo'])),
                                       columns=['most_similar_master_id', 'most_similar_master']
                                       ).astype(dtype= {"most_similar_master_id":"Int64",
        "most_similar_master":"str"})
        pd.testing.assert_frame_equal(expected_result, result)

    def test_get_groups_2_string_series_with_numeric_indexes_and_missing_master_value(self):
        """Should return a pd.DataFrame object with the length of the dupes. The series will contain the master string
        that matches the dupe with the highest similarity"""
        test_series_1 = pd.Series(['foooo', 'bar', 'foooo'], index = pd.Index([0, 1, 2], dtype = "Int64"))
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'], index = pd.Index([100, 101, 102, 103], dtype = "Int64"))
        sg = StringGrouper(test_series_1, test_series_2, replace_na=True)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.DataFrame(list(zip([0, 1, 102, 0], ['foooo', 'bar', 'baz', 'foooo'])),
                                       columns=['most_similar_index', 'most_similar_master'],
                                       index=test_series_2.index).astype(dtype= {"most_similar_index":"Int64",
        "most_similar_master":"str"})
        pd.testing.assert_frame_equal(expected_result, result)

    def test_get_groups_two_df_same_similarity(self):
        """Should return a pd.Series object with the length of the dupes. If there are two dupes with the same
        similarity, the first one is chosen"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz', 'foooo'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        sg = StringGrouper(test_series_1, test_series_2, ignore_index=True)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.Series(['foooo', 'bar', 'baz', 'foooo'], name='most_similar_master')
        pd.testing.assert_series_equal(expected_result, result)

    def test_get_groups_4_df_same_similarity(self):
        """Should return a pd.DataFrame object with the length of the dupes. If there are two dupes with the same
        similarity, the first one is chosen"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz', 'foooo'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        test_series_id_1 = pd.Series(['A0', 'A1', 'A2', 'A3'])
        test_series_id_2 = pd.Series(['B0', 'B1', 'B2', 'B3'])
        sg = StringGrouper(test_series_1,
                           test_series_2,
                           master_id=test_series_id_1,
                           duplicates_id=test_series_id_2,
                           ignore_index=True)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.DataFrame(list(zip(['A0', 'A1', 'A2', 'A0'], ['foooo', 'bar', 'baz', 'foooo'])),
                                       columns=['most_similar_master_id', 'most_similar_master'])
        pd.testing.assert_frame_equal(expected_result, result)

    def test_get_groups_two_df_no_match(self):
        """Should return a pd.Series object with the length of the dupes. If no match is found in dupes,
        the original will be returned"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz'])
        test_series_2 = pd.Series(['foooo', 'dooz', 'bar', 'baz', 'foooob'])
        sg = StringGrouper(test_series_1, test_series_2, ignore_index=True)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.Series(['foooo', 'dooz', 'bar', 'baz', 'foooo'], name='most_similar_master')
        pd.testing.assert_series_equal(expected_result, result)

    def test_get_groups_4_df_no_match(self):
        """Should return a pd.DataFrame object with the length of the dupes. If no match is found in dupes,
        the original will be returned"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz'])
        test_series_2 = pd.Series(['foooo', 'dooz', 'bar', 'baz', 'foooob'])
        test_series_id_1 = pd.Series(['A0', 'A1', 'A2'])
        test_series_id_2 = pd.Series(['B0', 'B1', 'B2', 'B3', 'B4'])
        sg = StringGrouper(test_series_1,
                           test_series_2,
                           master_id=test_series_id_1,
                           duplicates_id=test_series_id_2,
                           ignore_index=True)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.DataFrame(list(zip(
                ['A0', 'B1', 'A1', 'A2', 'A0'], ['foooo', 'dooz', 'bar', 'baz', 'foooo']
            )),
            columns=['most_similar_master_id', 'most_similar_master']
        )
        pd.testing.assert_frame_equal(expected_result, result)

    def test_get_groups_raises_exception(self):
        """Should raise an exception if called before the StringGrouper is fit"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz', 'foooo'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        sg = StringGrouper(test_series_1, test_series_2)
        with self.assertRaises(StringGrouperNotFitException):
            _ = sg.get_groups()

    def test_add_match_raises_exception_if_string_not_present(self):
        test_series_1 = pd.Series(['foooo', 'no match', 'baz', 'foooo'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        sg = StringGrouper(test_series_1).fit()
        sg2 = StringGrouper(test_series_1, test_series_2).fit()
        with self.assertRaises(ValueError):
            sg.add_match('doesnt exist', 'baz')
        with self.assertRaises(ValueError):
            sg.add_match('baz', 'doesnt exist')
        with self.assertRaises(ValueError):
            sg2.add_match('doesnt exist', 'baz')
        with self.assertRaises(ValueError):
            sg2.add_match('baz', 'doesnt exist')

    def test_add_match_single_occurence(self):
        """Should add the match if there are no exact duplicates"""
        test_series_1 = pd.Series(['foooo', 'no match', 'baz', 'foooo'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        sg = StringGrouper(test_series_1).fit()
        sg.add_match('no match', 'baz')
        matches = sg.get_matches()
        matches = matches[(matches.left_side == 'no match') & (matches.right_side == 'baz')]
        self.assertEqual(1, matches.shape[0])
        sg2 = StringGrouper(test_series_1, test_series_2).fit()
        sg2.add_match('no match', 'bar')
        matches = sg2.get_matches()
        matches = matches[(matches.left_side == 'no match') & (matches.right_side == 'bar')]
        self.assertEqual(1, matches.shape[0])

    def test_add_match_single_group_matches_symmetric(self):
        """New matches that are added to a SG with only a master series should be symmetric"""
        test_series_1 = pd.Series(['foooo', 'no match', 'baz', 'foooo'])
        sg = StringGrouper(test_series_1).fit()
        sg.add_match('no match', 'baz')
        matches = sg.get_matches()
        matches_1 = matches[(matches.left_side == 'no match') & (matches.right_side == 'baz')]
        self.assertEqual(1, matches_1.shape[0])
        matches_2 = matches[(matches.left_side == 'baz') & (matches.right_side == 'no match')]
        self.assertEqual(1, matches_2.shape[0])

    def test_add_match_multiple_occurences(self):
        """Should add multiple matches if there are exact duplicates"""
        test_series_1 = pd.Series(['foooo', 'no match', 'baz', 'foooo'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        sg = StringGrouper(test_series_1, test_series_2).fit()
        sg.add_match('foooo', 'baz')
        matches = sg.get_matches()
        matches = matches[(matches.left_side == 'foooo') & (matches.right_side == 'baz')]
        self.assertEqual(2, matches.shape[0])

    def test_remove_match(self):
        """Should remove a match"""
        test_series_1 = pd.Series(['foooo', 'no match', 'baz', 'foooob'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        sg = StringGrouper(test_series_1).fit()
        sg.remove_match('foooo', 'foooob')
        matches = sg.get_matches()
        matches_1 = matches[(matches.left_side == 'foooo') & (matches.right_side == 'foooob')]
        # In the case of only a master series, the matches are recursive, so both variants are to be removed
        matches_2 = matches[(matches.left_side == 'foooob') & (matches.right_side == 'foooo')]
        self.assertEqual(0, matches_1.shape[0])
        self.assertEqual(0, matches_2.shape[0])

        sg2 = StringGrouper(test_series_1, test_series_2).fit()
        sg2.remove_match('foooo', 'foooob')
        matches = sg2.get_matches()
        matches = matches[(matches.left_side == 'foooo') & (matches.right_side == 'foooob')]
        self.assertEqual(0, matches.shape[0])

    def test_string_grouper_type_error(self):
        """StringGrouper should raise an typeerror master or duplicates are not a series of strings"""
        with self.assertRaises(TypeError):
            _ = StringGrouper('foo', 'bar')
        with self.assertRaises(TypeError):
            _ = StringGrouper(pd.Series(['foo', 'bar']), pd.Series(['foo', 1]))
        with self.assertRaises(TypeError):
            _ = StringGrouper(pd.Series(['foo', np.nan]), pd.Series(['foo', 'j']))

    def test_prior_matches_added(self):
        """When a new match is added, any pre-existing matches should also be updated"""
        sample = [
            'microsoftoffice 365 home',
            'microsoftoffice 365 pers',
            'microsoft office'
            ]

        df = pd.DataFrame(sample, columns=['name'])

        sg = StringGrouper(df['name'], ignore_index=True)
        sg = sg.fit()

        sg = sg.add_match('microsoft office', 'microsoftoffice 365 home')
        sg = sg.add_match('microsoftoffice 365 pers', 'microsoft office')
        df['deduped'] = sg.get_groups()
        # All strings should now match to the same "master" string
        self.assertEqual(1, len(df.deduped.unique()))


class SpMatmulRsEquivalenceTest(unittest.TestCase):
    """Tests that the sp_matmul_rs backend (use_sp_matmul_rs=True) yields the same results
    as the legacy sparse_dot_topn backend (use_sp_matmul_rs=False)"""

    def test_match_strings_single_series(self):
        """match_strings on a single Series (self-join) should be backend-independent"""
        simple_example = SimpleExample()
        df1 = simple_example.customers_df2['Customer Name']
        matches_rs = fix_row_order(
            match_strings(df1, min_similarity=0.1, use_sp_matmul_rs=True))
        matches_legacy = fix_row_order(
            match_strings(df1, min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches_legacy, matches_rs)

    def test_match_strings_two_series(self):
        """match_strings on two Series should be backend-independent"""
        simple_example = SimpleExample()
        df1 = simple_example.customers_df['Customer Name']
        df2 = simple_example.customers_df2['Customer Name']
        matches_rs = fix_row_order(
            match_strings(df1, df2, min_similarity=0.1, use_sp_matmul_rs=True))
        matches_legacy = fix_row_order(
            match_strings(df1, df2, min_similarity=0.1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches_legacy, matches_rs)

    def test_match_strings_with_ids(self):
        """match_strings with master_id and duplicates_id should be backend-independent"""
        simple_example = SimpleExample()
        matches_rs = fix_row_order(
            match_strings(simple_example.customers_df['Customer Name'],
                          simple_example.customers_df2['Customer Name'],
                          master_id=simple_example.customers_df['Customer ID'],
                          duplicates_id=simple_example.customers_df2['Customer ID'],
                          min_similarity=0.1,
                          use_sp_matmul_rs=True))
        matches_legacy = fix_row_order(
            match_strings(simple_example.customers_df['Customer Name'],
                          simple_example.customers_df2['Customer Name'],
                          master_id=simple_example.customers_df['Customer ID'],
                          duplicates_id=simple_example.customers_df2['Customer ID'],
                          min_similarity=0.1,
                          use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(matches_legacy, matches_rs)

    def test_match_most_similar(self):
        """match_most_similar should be backend-independent"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        result_rs = match_most_similar(test_series_1, test_series_2,
                                       ignore_index=True, use_sp_matmul_rs=True)
        result_legacy = match_most_similar(test_series_1, test_series_2,
                                           ignore_index=True, use_sp_matmul_rs=False)
        pd.testing.assert_series_equal(result_legacy, result_rs)

    def test_group_similar_strings(self):
        """group_similar_strings should be backend-independent"""
        simple_example = SimpleExample()
        df1 = simple_example.customers_df['Customer Name']
        result_rs = group_similar_strings(df1, min_similarity=0.6, ignore_index=True,
                                          use_sp_matmul_rs=True)
        result_legacy = group_similar_strings(df1, min_similarity=0.6, ignore_index=True,
                                              use_sp_matmul_rs=False)
        pd.testing.assert_series_equal(result_legacy, result_rs)
        # sanity-check against the known expected grouping
        pd.testing.assert_series_equal(simple_example.expected_result_centroid, result_rs)

    def test_zero_min_similarity(self):
        """zero-similarity matches should be included by both backends when min_similarity <= 0"""
        simple_example = SimpleExample()
        s_master = simple_example.customers_df['Customer Name']
        s_dup = simple_example.whatever_series_1
        matches_rs = match_strings(s_master, s_dup, min_similarity=0, use_sp_matmul_rs=True)
        matches_legacy = match_strings(s_master, s_dup, min_similarity=0, use_sp_matmul_rs=False)
        pd.testing.assert_frame_equal(matches_legacy, matches_rs)
        pd.testing.assert_frame_equal(simple_example.expected_result_with_zeroes, matches_rs)

    def test_chunk_cols_result_invariant(self):
        """chunk_cols is an sp_matmul_rs performance knob only: any value must yield identical results"""
        df1 = SimpleExample().customers_df2['Customer Name']
        base = fix_row_order(match_strings(df1, min_similarity=0.1))
        for chunk_cols in (1, 7, 64, 100000):
            tuned = fix_row_order(match_strings(df1, min_similarity=0.1, chunk_cols=chunk_cols))
            pd.testing.assert_frame_equal(base, tuned)

    def test_chunk_cols_validation(self):
        """chunk_cols must be None or a positive int; the legacy backend ignores it with a warning"""
        df1 = SimpleExample().customers_df2['Customer Name']
        # chunk_cols is ignored (with a warning) by the sparse_dot_topn backend
        base = fix_row_order(
            match_strings(df1, min_similarity=0.1, use_sp_matmul_rs=False, n_blocks=(1, 1)))
        ignored = fix_row_order(
            match_strings(df1, min_similarity=0.1, use_sp_matmul_rs=False, n_blocks=(1, 1), chunk_cols=64))
        pd.testing.assert_frame_equal(base, ignored)
        # chunk_cols must be a positive integer (bools are not integers here)
        for bad in (0, -1, 2.5, 'x', True):
            with self.assertRaises(Exception):
                match_strings(df1, min_similarity=0.1, chunk_cols=bad)
        # numpy integers are valid
        rs_base = fix_row_order(match_strings(df1, min_similarity=0.1))
        np_tuned = fix_row_order(match_strings(df1, min_similarity=0.1, chunk_cols=np.int64(64)))
        pd.testing.assert_frame_equal(rs_base, np_tuned)

    def test_rs_backend_retries_with_64bit_indices_on_overflow(self):
        """An OverflowError (32-bit index overflow) must be recovered by retrying sp_matmul_rs
        with idx_dtype=np.int64, staying on the fast backend rather than falling back"""
        df1 = SimpleExample().customers_df2['Customer Name']
        expected = fix_row_order(match_strings(df1, min_similarity=0.1, use_sp_matmul_rs=False))

        original_build = StringGrouper._build_matches_rs
        idx_dtypes = []

        def flaky(self, master_matrix, duplicate_matrix, idx_dtype=None):
            idx_dtypes.append(idx_dtype)
            if len(idx_dtypes) == 1:  # first (32-bit) attempt overflows
                raise OverflowError
            return original_build(self, master_matrix, duplicate_matrix, idx_dtype=idx_dtype)

        with patch.object(StringGrouper, '_build_matches_rs', autospec=True, side_effect=flaky):
            result = fix_row_order(match_strings(df1, min_similarity=0.1))
        pd.testing.assert_frame_equal(expected, result)
        # the first attempt used default (32-bit) indices, the retry used 64-bit indices
        self.assertEqual(idx_dtypes, [None, np.int64])

    def test_rs_backend_falls_back_when_recovery_exhausted(self):
        """fit() must fall back to the blocked sparse_dot_topn path when sp_matmul_rs keeps failing
        (OverflowError even at 64-bit) or runs out of memory, yielding the same results"""
        df1 = SimpleExample().customers_df2['Customer Name']
        expected = fix_row_order(match_strings(df1, min_similarity=0.1, use_sp_matmul_rs=False))
        for error in (OverflowError, MemoryError):
            with patch.object(StringGrouper, '_build_matches_rs', side_effect=error):
                fallback = fix_row_order(match_strings(df1, min_similarity=0.1))
            pd.testing.assert_frame_equal(expected, fallback)

    def test_n_blocks_ignored_by_rs_backend(self):
        """n_blocks is ignored (with a warning) by the sp_matmul_rs backend instead of raising,
        so pre-0.8 code that tunes n_blocks keeps working under the new default backend"""
        df1 = SimpleExample().customers_df2['Customer Name']
        base = fix_row_order(match_strings(df1, min_similarity=0.1))
        with_n_blocks = fix_row_order(match_strings(df1, min_similarity=0.1, n_blocks=(1, 2)))
        pd.testing.assert_frame_equal(base, with_n_blocks)
        # malformed n_blocks values are still rejected, regardless of backend
        with self.assertRaises(Exception):
            match_strings(df1, min_similarity=0.1, n_blocks=(0, 2))

    def test_backend_switch_on_reused_instance(self):
        """switching backends via the instance methods must not trip validators on options
        carried over from earlier calls (n_blocks / chunk_cols of the other backend)"""
        df1 = SimpleExample().customers_df2['Customer Name']
        sg = StringGrouper(df1, min_similarity=0.1)
        legacy = fix_row_order(sg.match_strings(df1, use_sp_matmul_rs=False, n_blocks=(1, 2)))
        # stale n_blocks from the previous call must not raise under the rs backend
        rs = fix_row_order(sg.match_strings(df1, use_sp_matmul_rs=True))
        pd.testing.assert_frame_equal(legacy, rs)
        # and stale chunk_cols must not raise when switching back to the legacy backend
        sg2 = StringGrouper(df1, min_similarity=0.1, chunk_cols=64)
        back_to_legacy = fix_row_order(sg2.match_strings(df1, use_sp_matmul_rs=False))
        pd.testing.assert_frame_equal(legacy, back_to_legacy)

    def test_n_blocks_honored_via_instance_methods(self):
        """n_blocks passed through an instance method must reach the legacy matmul, not a stale
        value captured at construction time"""
        df1 = SimpleExample().customers_df2['Customer Name']
        sg = StringGrouper(df1, min_similarity=0.1, use_sp_matmul_rs=False)
        with patch.object(sg, '_build_matches', wraps=sg._build_matches) as spy:
            sg.match_strings(df1, use_sp_matmul_rs=False, n_blocks=(1, 2))
        self.assertEqual(spy.call_args.args[2], (1, 2))


if __name__ == '__main__':
    unittest.main()
