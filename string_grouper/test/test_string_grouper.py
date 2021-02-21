import unittest
import pandas as pd
import numpy as np
from scipy.sparse.csr import csr_matrix
from string_grouper.string_grouper import DEFAULT_MIN_SIMILARITY, \
    DEFAULT_MAX_N_MATCHES, DEFAULT_REGEX, \
    DEFAULT_NGRAM_SIZE, DEFAULT_N_PROCESSES, DEFAULT_IGNORE_CASE, \
    StringGrouperConfig, StringGrouper, StringGrouperNotFitException


class StringGrouperConfigTest(unittest.TestCase):
    def test_config_defaults(self):
        """Empty initialisation should set default values"""
        config = StringGrouperConfig()
        self.assertEqual(config.min_similarity, DEFAULT_MIN_SIMILARITY)
        self.assertEqual(config.max_n_matches, DEFAULT_MAX_N_MATCHES)
        self.assertEqual(config.regex, DEFAULT_REGEX)
        self.assertEqual(config.ngram_size, DEFAULT_NGRAM_SIZE)
        self.assertEqual(config.number_of_processes, DEFAULT_N_PROCESSES)
        self.assertEqual(config.ignore_case, DEFAULT_IGNORE_CASE)

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
    def test_n_grams_case_unchanged(self):
        """Should return all ngrams in a string with case"""
        test_series = pd.Series(pd.Series(['aa']))
        # Explicit do not ignore case
        sg = StringGrouper(test_series, ignore_case=False)
        expected_result = ['McD', 'cDo', 'Don', 'ona', 'nal', 'ald', 'lds']
        self.assertListEqual(expected_result, sg.n_grams('McDonalds'))

    def test_n_grams_ignore_case_to_lower(self):
        """Should return all case insensitive ngrams in a string"""
        test_series = pd.Series(pd.Series(['aa'])) 
        # Explicit ignore case
        sg = StringGrouper(test_series, ignore_case=True)
        expected_result = ['mcd', 'cdo', 'don', 'ona', 'nal', 'ald', 'lds']
        self.assertListEqual(expected_result, sg.n_grams('McDonalds'))

    def test_n_grams_ignore_case_to_lower_with_defaults(self):
        """Should return all case insensitive ngrams in a string"""
        test_series = pd.Series(pd.Series(['aa'])) 
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
        np.testing.assert_array_equal(expected_matches, sg._build_matches(master, dupe).toarray())

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
        pd.testing.assert_frame_equal(expected_df, sg._matches_list)

    def test_get_matches_two_dataframes(self):
        test_series_1 = pd.Series(['foo', 'bar', 'baz'])
        test_series_2 = pd.Series(['foo', 'bar', 'bop'])
        sg = StringGrouper(test_series_1, test_series_2).fit()
        left_side = ['foo', 'bar']
        right_side = ['foo', 'bar']
        similarity = [1.0, 1.0]
        expected_df = pd.DataFrame({'left_side': left_side, 'right_side': right_side, 'similarity': similarity})
        pd.testing.assert_frame_equal(expected_df, sg.get_matches())

    def test_get_matches_single(self):
        test_series_1 = pd.Series(['foo', 'bar', 'baz', 'foo'])
        sg = StringGrouper(test_series_1)
        sg = sg.fit()
        left_side = ['foo', 'foo', 'bar', 'baz', 'foo', 'foo']
        right_side = ['foo', 'foo', 'bar', 'baz', 'foo', 'foo']
        similarity = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        expected_df = pd.DataFrame({'left_side': left_side, 'right_side': right_side, 'similarity': similarity})
        pd.testing.assert_frame_equal(expected_df, sg.get_matches())

    def test_get_matches_1_series_1_id_series(self):
        test_series_1 = pd.Series(['foo', 'bar', 'baz', 'foo'])
        test_series_id_1 = pd.Series(['A0', 'A1', 'A2', 'A3'])
        sg = StringGrouper(test_series_1, master_id=test_series_id_1)
        sg = sg.fit()
        left_side = ['foo', 'foo', 'bar', 'baz', 'foo', 'foo']
        left_side_id = ['A0', 'A0', 'A1', 'A2', 'A3', 'A3']
        right_side = ['foo', 'foo', 'bar', 'baz', 'foo', 'foo']
        right_side_id = ['A3', 'A0', 'A1', 'A2', 'A3', 'A0']
        similarity = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        expected_df = pd.DataFrame({'left_side_id': left_side_id, 'left_side': left_side,
                                    'right_side_id': right_side_id, 'right_side': right_side, 'similarity': similarity})
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
        right_side = ['foo', 'bar']
        right_side_id = ['B0', 'B1']
        similarity = [1.0, 1.0]
        expected_df = pd.DataFrame({'left_side_id': left_side_id, 'left_side': left_side,
                                    'right_side_id': right_side_id, 'right_side': right_side, 'similarity': similarity})
        pd.testing.assert_frame_equal(expected_df, sg.get_matches())

    def test_get_matches_raises_exception_if_unexpected_options_given(self):
        # When the input id data does not correspond with its string data:
        test_series_1 = pd.Series(['foo', 'bar', 'baz'])
        test_series_id_1 = pd.Series(['A0', 'A1'])
        test_series_2 = pd.Series(['foo', 'bar', 'bop'])
        test_series_id_2 = pd.Series(['B0', 'B1'])
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, master_id=test_series_id_1)
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, duplicates=test_series_2, duplicates_id=test_series_id_2,
                              master_id=test_series_id_1)
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, duplicates=test_series_2, master_id=test_series_id_1)
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, test_series_2, duplicates_id=test_series_id_2)
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, duplicates_id=test_series_id_2)
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, duplicates_id=test_series_id_2, master_id=test_series_id_1)

        # When the input data is ok but the option combinations are invalid:
        test_series_1 = pd.Series(['foo', 'bar', 'baz'])
        test_series_id_1 = pd.Series(['A0', 'A1', 'A2'])
        test_series_2 = pd.Series(['foo', 'bar', 'bop'])
        test_series_id_2 = pd.Series(['B0', 'B1', 'B2'])
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, test_series_2, master_id=test_series_id_1)
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, test_series_2, duplicates_id=test_series_id_2)
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, duplicates_id=test_series_id_2)
        with self.assertRaises(Exception):
            _ = StringGrouper(test_series_1, master_id=test_series_id_1, duplicates_id=test_series_id_2)

    def test_get_groups_single_df(self):
        """Should return a pd.series object with the same length as the original df. The series object will contain
        a list of the grouped strings"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        sg = StringGrouper(test_series_1)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.Series(['foooo', 'bar', 'baz', 'foooo'])
        pd.testing.assert_series_equal(expected_result, result)

    def test_get_groups_1_string_series_1_id_series(self):
        """Should return a pd.series object with the same length as the original df. The series object will contain
        a list of the grouped strings"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        test_series_id_1 = pd.Series(['A0', 'A1', 'A2', 'A3'])
        sg = StringGrouper(test_series_1, master_id=test_series_id_1)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.DataFrame(list(zip(['A0', 'A1', 'A2', 'A0'], ['foooo', 'bar', 'baz', 'foooo'])))
        pd.testing.assert_frame_equal(expected_result, result)

    def test_get_groups_two_df(self):
        """Should return a pd.series object with the length of the dupes. The series will contain the master string
        that matches the dupe with the highest similarity"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        sg = StringGrouper(test_series_1, test_series_2)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.Series(['foooo', 'bar', 'baz', 'foooo'])
        pd.testing.assert_series_equal(expected_result, result)

    def test_get_groups_2_string_series_2_id_series(self):
        """Should return a pd.series object with the length of the dupes. The series will contain the master string
        that matches the dupe with the highest similarity"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        test_series_id_1 = pd.Series(['A0', 'A1', 'A2'])
        test_series_id_2 = pd.Series(['B0', 'B1', 'B2', 'B3'])
        sg = StringGrouper(test_series_1, test_series_2, master_id=test_series_id_1, duplicates_id=test_series_id_2)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.DataFrame(list(zip(['A0', 'A1', 'A2', 'A0'], ['foooo', 'bar', 'baz', 'foooo'])))
        pd.testing.assert_frame_equal(expected_result, result)

    def test_get_groups_two_df_same_similarity(self):
        """Should return a pd.series object with the length of the dupes. If there are two dupes with the same
        similarity, the first one is chosen"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz', 'foooo'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        sg = StringGrouper(test_series_1, test_series_2)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.Series(['foooo', 'bar', 'baz', 'foooo'])
        pd.testing.assert_series_equal(expected_result, result)

    def test_get_groups_4_df_same_similarity(self):
        """Should return a pd.series object with the length of the dupes. If there are two dupes with the same
        similarity, the first one is chosen"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz', 'foooo'])
        test_series_2 = pd.Series(['foooo', 'bar', 'baz', 'foooob'])
        test_series_id_1 = pd.Series(['A0', 'A1', 'A2', 'A3'])
        test_series_id_2 = pd.Series(['B0', 'B1', 'B2', 'B3'])
        sg = StringGrouper(test_series_1, test_series_2, master_id=test_series_id_1, duplicates_id=test_series_id_2)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.DataFrame(list(zip(['A0', 'A1', 'A2', 'A0'], ['foooo', 'bar', 'baz', 'foooo'])))
        pd.testing.assert_frame_equal(expected_result, result)

    def test_get_groups_two_df_no_match(self):
        """Should return a pd.series object with the length of the dupes. If no match is found in dupes,
        the original will be returned"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz'])
        test_series_2 = pd.Series(['foooo', 'dooz', 'bar', 'baz', 'foooob'])
        sg = StringGrouper(test_series_1, test_series_2)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.Series(['foooo', 'dooz', 'bar', 'baz', 'foooo'])
        pd.testing.assert_series_equal(expected_result, result)

    def test_get_groups_4_df_no_match(self):
        """Should return a pd.series object with the length of the dupes. If no match is found in dupes,
        the original will be returned"""
        test_series_1 = pd.Series(['foooo', 'bar', 'baz'])
        test_series_2 = pd.Series(['foooo', 'dooz', 'bar', 'baz', 'foooob'])
        test_series_id_1 = pd.Series(['A0', 'A1', 'A2'])
        test_series_id_2 = pd.Series(['B0', 'B1', 'B2', 'B3', 'B4'])
        sg = StringGrouper(test_series_1, test_series_2, master_id=test_series_id_1, duplicates_id=test_series_id_2)
        sg = sg.fit()
        result = sg.get_groups()
        expected_result = pd.DataFrame(list(zip(['A0', 'B1', 'A1', 'A2', 'A0'], ['foooo', 'dooz', 'bar', 'baz', 'foooo'])))
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

        sg = StringGrouper(df['name'])
        sg = sg.fit()

        sg = sg.add_match('microsoft office', 'microsoftoffice 365 home')
        sg = sg.add_match('microsoftoffice 365 pers', 'microsoft office')
        df['deduped'] = sg.get_groups()
        # All strings should now match to the same "master" string
        self.assertEqual(1, len(df.deduped.unique()))


if __name__ == '__main__':
    unittest.main()
