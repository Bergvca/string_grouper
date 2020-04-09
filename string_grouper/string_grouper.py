import pandas as pd
import numpy as np
import re
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.csr import csr_matrix
from typing import Tuple, NamedTuple, List, Optional
from sparse_dot_topn import awesome_cossim_topn
from functools import wraps


DEFAULT_NGRAM_SIZE: int = 3
DEFAULT_REGEX: str = r'[,-./]|\s'
DEFAULT_MAX_N_MATCHES: int = 20
DEFAULT_MIN_SIMILARITY: float = 0.8  # Minimum cosine similarity for an item to be considered a match
DEFAULT_N_PROCESSES: int = multiprocessing.cpu_count() - 1
DEFAULT_IGNORE_CASE: bool = True # ignores case by default

# High level functions

def group_similar_strings(strings_to_group: pd.Series, **kwargs) -> pd.Series:
    """
    Finds all similar strings in 'strings_to_group' and returns a Series of strings of the same length as
    strings_to_group. For each group of similar strings a single string is chosen as the 'master' string and is
    returned for each member of the group.

    For example the input series: [foooo, foooob, bar] will return [foooo, foooo, bar]
    Here 'foooo' and 'foooob' are grouped together into group 'foooo' because they are found to be very similar

    :param strings_to_group: pandas.Series. The input series of strings to be grouped
    :param kwargs: All other keyword arguments are passed to StringGrouperConfig
    :return: pandas.Series
    """
    string_grouper = StringGrouper(strings_to_group, **kwargs).fit()
    return string_grouper.get_groups()


def match_most_similar(master: pd.Series, duplicates: pd.Series, **kwargs) -> pd.Series:
    """
    Returns a series of strings of the same length as 'duplicates' where for each string in duplicates the most similar
    string in 'master' is returned. If there are no similar strings in master for a given string in duplicates
    (there is no potential match where the cosine similarity is above the threshold (default: 0.8))
    the original string in duplicates is returned.

    For example the input series [foooo, bar, baz] (master) and [foooob, bar, new] will return:
    [foooo, bar, new]

    :param master: pandas.Series. Series of strings that the duplicates will be matched with
    :param duplicates: pandas.Series. Series of strings that will me matched with the master
    :param kwargs: All other keyword arguments are passed to StringGrouperConfig
    :return: pandas.Series
    """
    string_grouper = StringGrouper(master, duplicates=duplicates, **kwargs).fit()
    return string_grouper.get_groups()


def match_strings(master: pd.Series, duplicates: Optional[pd.Series] = None, **kwargs) -> pd.DataFrame:
    """
    Returns all highly similar strings. If only 'master' is given, it will return highly similar strings within master.
    This can be seen as an self-join. If both master and duplicates is given, it will return highly similar strings
    between master and duplicates. This can be seen as an inner-join.

    :param master: pandas.Series. Series of strings against which matches are calculated
    :param duplicates: pandas.Series. Series of strings that will be matched with master if given (Optional)
    :param kwargs: All other keyword arguments are passed to StringGrouperConfig
    :return: pandas.Dataframe
    """
    string_grouper = StringGrouper(master, duplicates=duplicates, **kwargs).fit()
    return string_grouper.get_matches()


class StringGrouperConfig(NamedTuple):
    """
    Class with configuration variables

    :param ngram_size: int. The amount of characters in each n-gram. Optional. Default is 3
    :param regex: str. The regex string used to cleanup the input string. Optional. Default is [,-./]|\s
    :param max_n_matches: int. The maximum number of matches allowed per string. Default is 20
    :param min_similarity: float. The minium cossine similarity for two strings to be considered a match.
    Defaults to 0.8
    :param number_of_processes: int. The number of processes used by the cosine similarity calculation. Defaults to
    1 - number of cores on a machine.
    :param ignore_case: bool. Whether or not case should be ignored. Defaults to True (ignore case)
    """

    ngram_size: int = DEFAULT_NGRAM_SIZE
    regex: str = DEFAULT_REGEX
    max_n_matches: int = DEFAULT_MAX_N_MATCHES
    min_similarity: float = DEFAULT_MIN_SIMILARITY
    number_of_processes: int = DEFAULT_N_PROCESSES
    ignore_case: bool = DEFAULT_IGNORE_CASE


def validate_is_fit(f):
    """Validates if the StringBuilder was fit before calling certain public functions"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if args[0].is_build:
            return f(*args, **kwargs)
        else:
            raise StringGrouperNotFitException(f'{f.__name__} was called before the "fit" function was called.'
                                               f' Make sure to run fit the StringGrouper first using '
                                               f'StringGrouper.fit()')
    return wrapper


class StringGrouperNotFitException(Exception):
    """Raised when one of the public functions is called which requires the StringGrouper to be fit first"""
    pass


class StringGrouper(object):
    def __init__(self, master: pd.Series, duplicates: Optional[pd.Series] = None, **kwargs):
        """
        StringGrouper is a class that holds the matrix with cosine similarities between the master and duplicates
        matrix. If duplicates is not given it is replaced by master. To build this matrix the `fit` function must be
        called. It is possible to add and remove matches after building with the add_match and remove_match functions

        :param master: pandas.Series. A series of strings in which similar strings are searched, either against itself
        or against the `duplicates` series.
        :param duplicates: pandas.Series. If set, for each string in duplicates a similar string is searched in Master.
        :param kwargs: All other keyword arguments are passed to StringGrouperConfig
        """
        # Validate input
        if not StringGrouper._is_series_of_strings(master) or \
                (duplicates is not None and not StringGrouper._is_series_of_strings(duplicates)):
            raise TypeError('Input does not consist of pandas.Series containing only Strings')

        self._config: StringGrouperConfig = StringGrouperConfig(**kwargs)
        self._master: pd.Series = master.reset_index(drop=True)
        self._duplicates: pd.Series = duplicates.reset_index(drop=True) if duplicates is not None else None
        self.is_build = False  # indicates if the grouper was fit or not
        self._vectorizer = TfidfVectorizer(min_df=1, analyzer=self.n_grams)
        # After the StringGrouper is build, _matches_list will contain the indices and similarities of two matches
        self._matches_list: pd.DataFrame = pd.DataFrame()

    def n_grams(self, string: str) -> List[str]:
        """
        :param string: string to create ngrams from
        :return: list of ngrams
        """
        ngram_size = self._config.ngram_size
        regex_pattern = self._config.regex
        if (self._config.ignore_case and string is not None):
            string = string.lower() # lowercase to ignore all case
        string = re.sub(regex_pattern, r'', string)
        n_grams = zip(*[string[i:] for i in range(ngram_size)])
        return [''.join(n_gram) for n_gram in n_grams]

    def fit(self) -> 'StringGrouper':
        """Builds the _matches list which contains string matches indices and similarity"""
        master_matrix, duplicate_matrix = self._get_tf_idf_matrices()
        # Calculate the matches using the cosine similarity
        matches = self._build_matches(master_matrix, duplicate_matrix)
        # retrieve all matches
        self._matches_list = self._get_matches_list(matches)
        self.is_build = True
        return self

    @validate_is_fit
    def get_matches(self) -> pd.DataFrame:
        """Returns a DataFrame with all the matches and their cosine similarity"""
        left_side = self._master[self._matches_list.master_side].reset_index(drop=True)

        if self._duplicates is not None:
            right_side = self._duplicates[self._matches_list.dupe_side].reset_index(drop=True)
        else:
            right_side = self._master[self._matches_list.dupe_side].reset_index(drop=True)

        similarity = self._matches_list.similarity.reset_index(drop=True)
        return pd.DataFrame({
            'left_side': left_side,
            'right_side': right_side,
            'similarity': similarity
        })

    @validate_is_fit
    def get_groups(self) -> pd.Series:
        """If there is only a master series of strings, this will return the 'master' strings.
         A single string in a group of near duplicates is chosen for as 'master' and is returned for each string
         in the master series.
         If there is a master series and a duplicate series, the most similar master is picked
         for each duplicate and returned
         """
        if self._duplicates is None:
            return self._deduplicate()
        else:
            return self._get_nearest_matches()

    @validate_is_fit
    def add_match(self, master_side: str, dupe_side: str) -> 'StringGrouper':
        """Adds a match if it wasn't found by the fit function"""
        master_indices, dupe_indices = self._get_indices_of(master_side, dupe_side)
        similarities = [1]

        # cross join the indices
        new_matches = StringGrouper._cross_join(dupe_indices, master_indices, similarities)
        # If we are deduping within one Series, we need to make sure the matches stay symmetric
        if self._duplicates is None:
            new_matches = StringGrouper._make_symmetric(new_matches)
        # update the matches
        self._matches_list = pd.concat([self._matches_list, new_matches])
        return self

    @validate_is_fit
    def remove_match(self, master_side: str, dupe_side: str) -> 'StringGrouper':
        """ Removes a match from the StringGrouper"""
        master_indices, dupe_indices = self._get_indices_of(master_side, dupe_side)
        # In the case of having only a master series, we need to remove both the master - dupe match
        # and the dupe - master match:
        if self._duplicates is None:
            master_indices = pd.concat([master_indices, dupe_indices])
            dupe_indices = master_indices

        self._matches_list = self._matches_list[
            ~(
                    (self._matches_list.master_side.isin(master_indices)) &
                    (self._matches_list.dupe_side.isin(dupe_indices))
            )]
        return self

    def _get_tf_idf_matrices(self) -> Tuple[csr_matrix, csr_matrix]:
        # Fit the tf-idf vectorizer
        self._vectorizer = self._fit_vectorizer()
        # Build the two matrices
        master_matrix = self._vectorizer.transform(self._master)

        if self._duplicates is not None:
            duplicate_matrix = self._vectorizer.transform(self._duplicates)
        # IF there is no duplicate matrix, we assume we want to match on the master matrix itself
        else:
            duplicate_matrix = master_matrix

        return master_matrix, duplicate_matrix

    def _fit_vectorizer(self) -> TfidfVectorizer:
        # if both dupes and master string series are set - we concat them to fit the vectorizer on all
        # strings
        if self._duplicates is not None:
            strings = pd.concat([self._master, self._duplicates])
        else:
            strings = self._master
        self._vectorizer.fit(strings)
        return self._vectorizer

    def _build_matches(self, master_matrix: csr_matrix, duplicate_matrix: csr_matrix) -> csr_matrix:
        """Builds the cossine similarity matrix of two csr matrices"""
        tf_idf_matrix_1 = master_matrix
        tf_idf_matrix_2 = duplicate_matrix.transpose()

        optional_kwargs = dict()
        if self._config.number_of_processes > 1:
            optional_kwargs = {
                'use_threads': True,
                'n_jobs': self._config.number_of_processes
            }

        return awesome_cossim_topn(tf_idf_matrix_1, tf_idf_matrix_2,
                                   self._config.max_n_matches,
                                   self._config.min_similarity,
                                   **optional_kwargs)

    @staticmethod
    def _get_matches_list(matches) -> pd.DataFrame:
        """Returns a list of all the indices of matches"""
        non_zeros = matches.nonzero()

        sparserows = non_zeros[0]
        sparsecols = non_zeros[1]
        nr_matches = sparsecols.size
        master_side = np.empty([nr_matches], dtype=int)
        dupe_side = np.empty([nr_matches], dtype=int)
        similarity = np.zeros(nr_matches)

        for index in range(0, nr_matches):
            master_side[index] = sparserows[index]
            dupe_side[index] = sparsecols[index]
            similarity[index] = matches.data[index]

        matches_list = pd.DataFrame({'master_side': master_side,
                                     'dupe_side': dupe_side,
                                     'similarity': similarity})
        return matches_list

    @staticmethod
    def _clean_groups(grouped_id_tuples: pd.DataFrame) -> pd.DataFrame:
        """Clean groups by merging groups that have an item in between them with a high similarity"""
        # Find the groups where the min id is not equal to the group id
        id_tuples_min = grouped_id_tuples.groupby('group_id').agg('min').reset_index()
        orphans = id_tuples_min[id_tuples_min.group_id != id_tuples_min.original_id].copy()
        if orphans.shape[0] > 0:
            # Get the new group id's
            new_group_id = (orphans
                            .merge(grouped_id_tuples,
                                   left_on='group_id', right_on='original_id', suffixes=('_orig', '_new'))
                            [['group_id_orig', 'group_id_new']]
                            .drop_duplicates())
            # join them with the old group ids
            new_grouped_id_tuples = grouped_id_tuples.merge(new_group_id,
                                                            how='outer',
                                                            left_on='group_id', right_on='group_id_orig')
            # update the old ones
            rows_to_update = ~new_grouped_id_tuples.group_id_new.isnull()
            new_grouped_id_tuples.loc[rows_to_update, 'group_id'] = new_grouped_id_tuples[rows_to_update].group_id_new
            grouped_id_tuples = new_grouped_id_tuples[['original_id', 'group_id', 'min_similarity']].copy()
            grouped_id_tuples.group_id = grouped_id_tuples.group_id.astype('int64')
            # repeat if necessary
            return StringGrouper._clean_groups(grouped_id_tuples)
        else:
            return grouped_id_tuples

    def _get_nearest_matches(self) -> pd.Series:

        dupes = self._duplicates.rename('duplicates')
        master = self._master.rename('master')

        dupes_max_sim = self._matches_list.groupby('dupe_side').agg({'similarity': 'max'}).reset_index()
        dupes_max_sim = dupes_max_sim.merge(self._matches_list, on=['dupe_side', 'similarity'])

        # in case there are multiple equal similarities, we pick the one that comes first
        dupes_max_sim = dupes_max_sim.groupby(['dupe_side']).agg({'master_side': 'min'}).reset_index()

        # First we add the duplicate strings
        dupes_max_sim = dupes_max_sim.merge(dupes, left_on='dupe_side', right_index=True, how='outer')

        # Now add the master strings
        dupes_max_sim = dupes_max_sim.merge(master, left_on='master_side', right_index=True, how='left')

        # update the master series with the duplicates in cases were there is no match
        rows_to_update = dupes_max_sim.master.isnull()
        dupes_max_sim.loc[rows_to_update, 'master'] = dupes_max_sim[rows_to_update].duplicates
        # make sure to keep same order as duplicates
        dupes_max_sim = dupes_max_sim.sort_values('dupe_side').set_index('dupe_side')
        dupes_max_sim.index.rename(None, inplace=True)
        return dupes_max_sim['master'].rename(None)

    def _deduplicate(self) -> pd.Series:
        master_indices = self._master.index.to_series()
        index_to_index = pd.DataFrame({
            'master_side': master_indices,
            'dupe_side': master_indices,
            'similarity': np.full(master_indices.shape[0], 1)
        })
        all_id_tuples = pd.concat([self._matches_list, index_to_index])

        # get the groups
        grouped_id_tuples = all_id_tuples.groupby('dupe_side').agg('min').reset_index()
        grouped_id_tuples.columns = ['original_id', 'group_id', 'min_similarity']

        # clean the groups:
        grouped_id_tuples = StringGrouper._clean_groups(grouped_id_tuples)
        grouped_id_tuples = grouped_id_tuples.sort_values(by='original_id')

        # Get the strings belonging to the group ids
        group_id_strings = self._master[grouped_id_tuples.group_id].reset_index(drop=True)
        return group_id_strings

    def _get_indices_of(self, master_side: str, dupe_side: str) -> Tuple[pd.Series, pd.Series]:
        master_strings = self._master
        dupe_strings = self._master if self._duplicates is None else self._duplicates
        # Check if input is valid:
        self._validate_strings_exist(master_side, dupe_side, master_strings, dupe_strings)
        # Get the indices of the two strings
        master_indices = master_strings[master_strings == master_side].index.to_series().reset_index(drop=True)
        dupe_indices = dupe_strings[dupe_strings == dupe_side].index.to_series().reset_index(drop=True)
        return master_indices, dupe_indices

    @staticmethod
    def _make_symmetric(new_matches: pd.DataFrame) -> pd.DataFrame:
        columns_switched = pd.DataFrame({'master_side': new_matches.dupe_side,
                                         'dupe_side': new_matches.master_side,
                                         'similarity': new_matches.similarity})
        return pd.concat([new_matches, columns_switched])

    @staticmethod
    def _cross_join(dupe_indices, master_indices, similarities) -> pd.DataFrame:
        x_join_index = pd.MultiIndex.from_product([master_indices, dupe_indices, similarities],
                                                  names=['master_side', 'dupe_side', 'similarity'])
        x_joined_df = pd.DataFrame(index=x_join_index).reset_index()
        return x_joined_df

    @staticmethod
    def _validate_strings_exist(master_side, dupe_side, master_strings, dupe_strings):
        if not master_strings.isin([master_side]).any():
            raise ValueError(f'{master_side} not found in StringGrouper string series')
        elif not dupe_strings.isin([dupe_side]).any():
            raise ValueError(f'{dupe_side} not found in StringGrouper dupe string series')

    @staticmethod
    def _is_series_of_strings(series_to_test: pd.Series) -> bool:
        if not isinstance(series_to_test, pd.Series):
            return False
        elif series_to_test.str.len().isna().any():
            return False
        return True
