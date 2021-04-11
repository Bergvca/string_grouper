import pandas as pd
import numpy as np
import re
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.csr import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import Tuple, NamedTuple, List, Optional, Union
from sparse_dot_topn import awesome_cossim_topn
from functools import wraps
import warnings

DEFAULT_NGRAM_SIZE: int = 3
DEFAULT_REGEX: str = r'[,-./]|\s'
DEFAULT_MAX_N_MATCHES: int = 20
DEFAULT_MIN_SIMILARITY: float = 0.8  # minimum cosine similarity for an item to be considered a match
DEFAULT_N_PROCESSES: int = multiprocessing.cpu_count() - 1
DEFAULT_IGNORE_CASE: bool = True  # ignores case by default
DEFAULT_DROP_INDEX: bool = False  # includes index-columns in output
DEFAULT_REPLACE_NA: bool = False    # when finding the most similar strings, does not replace NaN values in most
                                    # similar string index-columns with corresponding duplicates-index values
DEFAULT_INCLUDE_ZEROES: bool = True # when the minimum cosine similarity <=0, determines whether zero-similarity
                                    # matches appear in the output 
DEFAULT_SUPPRESS_WARNING: bool = False  # when the minimum cosine similarity <=0 and zero-similarity matches are
                                        # requested, determines whether or not to suppress the message warning that 
                                        # max_n_matches may be too small 
GROUP_REP_CENTROID: str = 'centroid'    # Option value to select the string in each group with the largest
                                        # similarity aggregate as group-representative:
GROUP_REP_FIRST: str = 'first'  # Option value to select the first string in each group as group-representative:
DEFAULT_GROUP_REP: str = GROUP_REP_CENTROID # chooses group centroid as group-representative by default

# The following string constants are used by (but aren't [yet] options passed to) StringGrouper
DEFAULT_COLUMN_NAME: str = 'side'   # used to name non-index columns of the output of StringGrouper.get_matches
DEFAULT_ID_NAME: str = 'id' # used to name id-columns in the output of StringGrouper.get_matches
LEFT_PREFIX: str = 'left_'  # used to prefix columns on the left of the output of StringGrouper.get_matches
RIGHT_PREFIX: str = 'right_'    # used to prefix columns on the right of the output of StringGrouper.get_matches
MOST_SIMILAR_PREFIX: str = 'most_similar_'  # used to prefix columns of the output of
                                            # StringGrouper._get_nearest_matches
DEFAULT_MASTER_NAME: str = 'master' # used to name non-index column of the output of StringGrouper.get_nearest_matches
DEFAULT_MASTER_ID_NAME: str = f'{DEFAULT_MASTER_NAME}_{DEFAULT_ID_NAME}'    # used to name id-column of the output of
                                                                            # StringGrouper.get_nearest_matches
GROUP_REP_PREFIX: str = 'group_rep_'    # used to prefix and name columns of the output of StringGrouper._deduplicate

# High level functions


def compute_pairwise_similarities(string_series_1: pd.Series,
                                  string_series_2: pd.Series,
                                  **kwargs) -> pd.Series:
    """
    Computes the similarity scores between two Series of strings row-wise.

    :param string_series_1: pandas.Series. The input Series of strings to be grouped
    :param string_series_2: pandas.Series. The input Series of the IDs of the strings to be grouped
    :param kwargs: All other keyword arguments are passed to StringGrouperConfig
    :return: pandas.Series of similarity scores, the same length as string_series_1 and string_series_2
    """
    return StringGrouper(string_series_1, string_series_2, **kwargs).dot()


def group_similar_strings(strings_to_group: pd.Series,
                          string_ids: Optional[pd.Series] = None,
                          **kwargs) -> Union[pd.DataFrame, pd.Series]:
    """
    If 'string_ids' is not given, finds all similar strings in 'strings_to_group' and returns a Series of
    strings of the same length as 'strings_to_group'. For each group of similar strings a single string
    is chosen as the 'master' string and is returned for each member of the group.

    For example the input Series: [foooo, foooob, bar] will return [foooo, foooo, bar].  Here 'foooo' and
    'foooob' are grouped together into group 'foooo' because they are found to be very similar.

    If string_ids is also given, a DataFrame of the strings and their corresponding IDs is instead returned.

    :param strings_to_group: pandas.Series. The input Series of strings to be grouped.
    :param string_ids: pandas.Series. The input Series of the IDs of the strings to be grouped. (Optional)
    :param kwargs: All other keyword arguments are passed to StringGrouperConfig. (Optional)
    :return: pandas.Series or pandas.DataFrame.
    """
    string_grouper = StringGrouper(strings_to_group, master_id=string_ids, **kwargs).fit()
    return string_grouper.get_groups()


def match_most_similar(master: pd.Series,
                       duplicates: pd.Series,
                       master_id: Optional[pd.Series] = None,
                       duplicates_id: Optional[pd.Series] = None,
                       **kwargs) -> Union[pd.DataFrame, pd.Series]:
    """
    If no IDs ('master_id' and 'duplicates_id') are given, returns a Series of strings of the same length
    as 'duplicates' where for each string in duplicates the most similar string in 'master' is returned.
    If there are no similar strings in master for a given string in duplicates
    (there is no potential match where the cosine similarity is above the threshold [default: 0.8])
    the original string in duplicates is returned.

    For example the input Series [foooo, bar, baz] (master) and [foooob, bar, new] will return:
    [foooo, bar, new].

    If IDs (both 'master_id' and 'duplicates_id') are also given, returns a DataFrame of the same strings
    output in the above case with their corresponding IDs.

    :param master: pandas.Series. Series of strings that the duplicates will be matched with.
    :param duplicates: pandas.Series. Series of strings that will me matched with the master.
    :param master_id: pandas.Series. Series of values that are IDs for master column rows. (Optional)
    :param duplicates_id: pandas.Series. Series of values that are IDs for duplicates column rows. (Optional)
    :param kwargs: All other keyword arguments are passed to StringGrouperConfig. (Optional)
    :return: pandas.Series or pandas.DataFrame.
    """
    string_grouper = StringGrouper(master,
                                   duplicates=duplicates,
                                   master_id=master_id,
                                   duplicates_id=duplicates_id,
                                   **kwargs).fit()
    return string_grouper.get_groups()


def match_strings(master: pd.Series,
                  duplicates: Optional[pd.Series] = None,
                  master_id: Optional[pd.Series] = None,
                  duplicates_id: Optional[pd.Series] = None,
                  **kwargs) -> pd.DataFrame:
    """
    Returns all highly similar strings. If only 'master' is given, it will return highly similar strings within master.
    This can be seen as an self-join. If both master and duplicates is given, it will return highly similar strings
    between master and duplicates. This can be seen as an inner-join.

    :param master: pandas.Series. Series of strings against which matches are calculated.
    :param duplicates: pandas.Series. Series of strings that will be matched with master if given (Optional).
    :param master_id: pandas.Series. Series of values that are IDs for master column rows (Optional).
    :param duplicates_id: pandas.Series. Series of values that are IDs for duplicates column rows (Optional).
    :param kwargs: All other keyword arguments are passed to StringGrouperConfig.
    :return: pandas.Dataframe.
    """
    string_grouper = StringGrouper(master,
                                   duplicates=duplicates,
                                   master_id=master_id,
                                   duplicates_id=duplicates_id,
                                   **kwargs).fit()
    return string_grouper.get_matches()


class StringGrouperConfig(NamedTuple):
    """
    Class with configuration variables.

    :param ngram_size: int. The amount of characters in each n-gram. Default is 3.
    :param regex: str. The regex string used to cleanup the input string. Default is [,-./]|\s.
    :param max_n_matches: int. The maximum number of matches allowed per string. Default is 20.
    :param min_similarity: float. The minimum cosine similarity for two strings to be considered a match.
    Defaults to 0.8.
    :param number_of_processes: int. The number of processes used by the cosine similarity calculation.
    Defaults to number of cores on a machine - 1.
    :param ignore_case: bool. Whether or not case should be ignored. Defaults to True (ignore case).
    :param ignore_index: whether or not to exclude string Series index-columns in output.  Defaults to False.
    :param include_zeroes: when the minimum cosine similarity <=0, determines whether zero-similarity matches 
    appear in the output.  Defaults to True.
    :param suppress_warning: when min_similarity <=0 and include_zeroes=True, determines whether or not to supress
    the message warning that max_n_matches may be too small.  Defaults to False.
    :param replace_na: whether or not to replace NaN values in most similar string index-columns with 
    corresponding duplicates-index values. Defaults to False.
    :param group_rep: str.  The scheme to select the group-representative.  Default is 'centroid'.
    The other choice is 'first'.
    """

    ngram_size: int = DEFAULT_NGRAM_SIZE
    regex: str = DEFAULT_REGEX
    max_n_matches: int = DEFAULT_MAX_N_MATCHES
    min_similarity: float = DEFAULT_MIN_SIMILARITY
    number_of_processes: int = DEFAULT_N_PROCESSES
    ignore_case: bool = DEFAULT_IGNORE_CASE
    ignore_index: bool = DEFAULT_DROP_INDEX
    include_zeroes: bool = DEFAULT_INCLUDE_ZEROES
    suppress_warning: bool = DEFAULT_SUPPRESS_WARNING
    replace_na: bool = DEFAULT_REPLACE_NA
    group_rep: str = DEFAULT_GROUP_REP


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
    def __init__(self, master: pd.Series,
                 duplicates: Optional[pd.Series] = None,
                 master_id: Optional[pd.Series] = None,
                 duplicates_id: Optional[pd.Series] = None,
                 **kwargs):
        """
        StringGrouper is a class that holds the matrix with cosine similarities between the master and duplicates
        matrix. If duplicates is not given it is replaced by master. To build this matrix the `fit` function must be
        called. It is possible to add and remove matches after building with the add_match and remove_match functions

        :param master: pandas.Series. A Series of strings in which similar strings are searched, either against itself
        or against the `duplicates` Series.
        :param duplicates: pandas.Series. If set, for each string in duplicates a similar string is searched in Master.
        :param master_id: pandas.Series. If set, contains ID values for each row in master Series.
        :param duplicates_id: pandas.Series. If set, contains ID values for each row in duplicates Series.
        :param kwargs: All other keyword arguments are passed to StringGrouperConfig
        """
        # Validate match strings input
        if not StringGrouper._is_series_of_strings(master) or \
                (duplicates is not None and not StringGrouper._is_series_of_strings(duplicates)):
            raise TypeError('Input does not consist of pandas.Series containing only Strings')
        # Validate optional IDs input
        if not StringGrouper._is_input_data_combination_valid(duplicates, master_id, duplicates_id):
            raise Exception('List of data Series options is invalid')
        StringGrouper._validate_id_data(master, duplicates, master_id, duplicates_id)

        self._master: pd.Series = master
        self._duplicates: pd.Series = duplicates if duplicates is not None else None
        self._master_id: pd.Series = master_id if master_id is not None else None
        self._duplicates_id: pd.Series = duplicates_id if duplicates_id is not None else None
        self._config: StringGrouperConfig = StringGrouperConfig(**kwargs)
        self._validate_group_rep_specs()
        self._validate_replace_na_and_drop()
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
        if self._config.ignore_case and string is not None:
            string = string.lower()  # lowercase to ignore all case
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
        if self._duplicates is None:
            # the list of matches needs to be symmetric!!! (i.e., if A != B and A matches B; then B matches A)
            self._symmetrize_matches_list()
        self.is_build = True
        return self

    def dot(self) -> pd.Series:
        """Computes the row-wise similarity scores between strings in _master and _duplicates"""
        if len(self._master) != len(self._duplicates):
            raise Exception("To perform this function, both input Series must have the same length.")
        master_matrix, duplicate_matrix = self._get_tf_idf_matrices()
        # Calculate pairwise cosine similarities:
        pairwise_similarities = np.asarray(master_matrix.multiply(duplicate_matrix).sum(axis=1)).squeeze()
        return pd.Series(pairwise_similarities, name='similarity', index=self._master.index)

    @validate_is_fit
    def get_matches(self,
                    ignore_index: Optional[bool] = None,
                    include_zeroes: Optional[bool]=None,
                    suppress_warning: Optional[bool]=None) -> pd.DataFrame:
        """
        Returns a DataFrame with all the matches and their cosine similarity.
        If optional IDs are used, returned as extra columns with IDs matched to respective data rows

        :param ignore_index: whether or not to exclude string Series index-columns in output.  Defaults to 
        self._config.ignore_index.
        :param include_zeroes: when the minimum cosine similarity <=0, determines whether zero-similarity matches 
        appear in the output.  Defaults to self._config.include_zeroes.
        :param suppress_warning: when min_similarity <=0 and include_zeroes=True, determines whether or not to suppress
        the message warning that max_n_matches may be too small.  Defaults to self._config.suppress_warning.
        """
        def get_both_sides(master: pd.Series,
                           duplicates: pd.Series,
                           generic_name=(DEFAULT_COLUMN_NAME, DEFAULT_COLUMN_NAME),
                           drop_index=False):
            lname, rname = generic_name
            left = master if master.name else master.rename(lname)
            left = left.iloc[matches_list.master_side].reset_index(drop=drop_index)
            if self._duplicates is None:
                right = master if master.name else master.rename(rname)
            else:
                right = duplicates if duplicates.name else duplicates.rename(rname)
            right = right.iloc[matches_list.dupe_side].reset_index(drop=drop_index)
            return left, (right if isinstance(right, pd.Series) else right[right.columns[::-1]])

        def prefix_column_names(data: Union[pd.Series, pd.DataFrame], prefix: str):
            if isinstance(data, pd.DataFrame):
                return data.rename(columns={c: f"{prefix}{c}" for c in data.columns})
            else:
                return data.rename(f"{prefix}{data.name}")

        if ignore_index is None: ignore_index = self._config.ignore_index
        if include_zeroes is None: include_zeroes = self._config.include_zeroes
        if suppress_warning is None: suppress_warning = self._config.suppress_warning
        if self._config.min_similarity > 0 or not include_zeroes:
            matches_list = self._matches_list
        elif include_zeroes:
            # Here's a fix to a bug pointed out by one GitHub user (@nbcvijanovic):
            # the fix includes zero-similarity matches that are missing by default 
            # in _matches_list due to our use of sparse matrices 
            non_matches_list = self._get_non_matches_list(suppress_warning)
            matches_list = self._matches_list if non_matches_list.empty else \
                pd.concat([self._matches_list, non_matches_list], axis=0, ignore_index=True)
            
        left_side, right_side = get_both_sides(self._master, self._duplicates, drop_index=ignore_index)
        similarity = matches_list.similarity.reset_index(drop=True)
        if self._master_id is None:
            return pd.concat(
                [
                    prefix_column_names(left_side, LEFT_PREFIX),
                    similarity,
                    prefix_column_names(right_side, RIGHT_PREFIX)
                ],
                axis=1
            )
        else:
            left_side_id, right_side_id = get_both_sides(
                self._master_id,
                self._duplicates_id,
                (DEFAULT_ID_NAME, DEFAULT_ID_NAME),
                drop_index=True
            )
            return pd.concat(
                [
                    prefix_column_names(left_side, LEFT_PREFIX),
                    prefix_column_names(left_side_id, LEFT_PREFIX),
                    similarity,
                    prefix_column_names(right_side_id, RIGHT_PREFIX),
                    prefix_column_names(right_side, RIGHT_PREFIX)
                ],
                axis=1
            )

    @validate_is_fit
    def get_groups(self,
                   ignore_index: Optional[bool] = None,
                   replace_na: Optional[bool] = None) -> Union[pd.DataFrame, pd.Series]:
        """If there is only a master Series of strings, this will return a Series of 'master' strings.
         A single string in a group of near duplicates is chosen as 'master' and is returned for each string
         in the master Series.
         If there is a master Series and a duplicate Series, the most similar master is picked
         for each duplicate and returned.
         If there are IDs (master_id and/or duplicates_id) then the IDs corresponding to the string outputs
         above are returned as well altogether in a DataFrame.

        :param ignore_index: whether or not to exclude string Series index-columns in output.  Defaults to 
        self._config.ignore_index.
        :param replace_na: whether or not to replace NaN values in most similar string index-columns with 
        corresponding duplicates-index values. Defaults to self._config.replace_na.
         """
        if ignore_index is None: ignore_index = self._config.ignore_index
        if self._duplicates is None:
            return self._deduplicate(ignore_index=ignore_index)
        else:
            if replace_na is None: replace_na = self._config.replace_na
            return self._get_nearest_matches(ignore_index=ignore_index, replace_na=replace_na)

    @validate_is_fit
    def add_match(self, master_side: str, dupe_side: str) -> 'StringGrouper':
        """Adds a match if it wasn't found by the fit function"""
        master_indices, dupe_indices = self._get_indices_of(master_side, dupe_side)

        # add prior matches to new match
        prior_matches = self._matches_list.master_side[self._matches_list.dupe_side.isin(dupe_indices)]
        dupe_indices = dupe_indices.append(prior_matches)
        dupe_indices.drop_duplicates(inplace=True)

        similarities = [1]

        # cross join the indices
        new_matches = StringGrouper._cross_join(dupe_indices, master_indices, similarities)
        # If we are de-duping within one Series, we need to make sure the matches stay symmetric
        if self._duplicates is None:
            new_matches = StringGrouper._make_symmetric(new_matches)
        # update the matches
        self._matches_list = pd.concat([self._matches_list.drop_duplicates(), new_matches], ignore_index=True)

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

    def _symmetrize_matches_list(self):
        # [symmetrized matches_list] = [matches_list] UNION [transposed matches_list] (i.e., column-names swapped):
        self._matches_list = self._matches_list.set_index(['master_side', 'dupe_side'])\
            .combine_first(
                self._matches_list.rename(
                    columns={
                        'master_side': 'dupe_side',
                        'dupe_side': 'master_side'
                    }
                ).set_index(['master_side', 'dupe_side'])
            ).reset_index()

    def _get_non_matches_list(self, suppress_warning=False) -> pd.DataFrame:
        """Returns a list of all the indices of non-matching pairs (with similarity set to 0)"""
        m_sz, d_sz = len(self._master), len(self._master if self._duplicates is None else self._duplicates)
        all_pairs = pd.MultiIndex.from_product([range(m_sz), range(d_sz)], names=['master_side', 'dupe_side'])
        matched_pairs = pd.MultiIndex.from_frame(self._matches_list[['master_side', 'dupe_side']])
        missing_pairs = all_pairs.difference(matched_pairs)
        if missing_pairs.empty: return pd.DataFrame()
        if (self._config.max_n_matches < d_sz) and not suppress_warning:
            warnings.warn(f'WARNING: max_n_matches={self._config.max_n_matches} may be too small!\n'
                          f'\t\t Some zero-similarity matches returned may be false!\n'
                          f'\t\t To be absolutely certain all zero-similarity matches are true,\n'
                          f'\t\t try setting max_n_matches={d_sz} (the length of the Series parameter duplicates).\n'
                          f'\t\t To suppress this warning, set suppress_warning=True.')
        missing_pairs = missing_pairs.to_frame(index=False)
        missing_pairs['similarity'] = 0
        return missing_pairs

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

    def _get_nearest_matches(self,
                             ignore_index=False,
                             replace_na=False) -> Union[pd.DataFrame, pd.Series]:
        prefix = MOST_SIMILAR_PREFIX
        master_label = f'{prefix}{self._master.name if self._master.name else DEFAULT_MASTER_NAME}'
        master = self._master.rename(master_label).reset_index(drop=ignore_index)
        dupes = self._duplicates.rename('duplicates').reset_index(drop=ignore_index)
        
        # Rename new master-columns to avoid possible conflict with new dupes-columns when later merging 
        if isinstance(dupes, pd.DataFrame):
            master.rename(
                columns={col: f'{prefix}{col}' for col in master.columns if str(col) != master_label},
                inplace=True
            )

        if self._master_id is not None:
            master_id_label = f'{prefix}{self._master_id.name if self._master_id.name else DEFAULT_MASTER_ID_NAME}'
            master = pd.concat([master, self._master_id.rename(master_id_label).reset_index(drop=True)], axis=1)
            dupes = pd.concat([dupes, self._duplicates_id.rename('duplicates_id').reset_index(drop=True)], axis=1)

        dupes_max_sim = self._matches_list.groupby('dupe_side').agg({'similarity': 'max'}).reset_index()
        dupes_max_sim = dupes_max_sim.merge(self._matches_list, on=['dupe_side', 'similarity'])

        # In case there are multiple equal similarities, we pick the one that comes first
        dupes_max_sim = dupes_max_sim.groupby(['dupe_side']).agg({'master_side': 'min'}).reset_index()

        # First we add the duplicate strings
        dupes_max_sim = dupes_max_sim.merge(dupes, left_on='dupe_side', right_index=True, how='outer')

        # Now add the master strings
        dupes_max_sim = dupes_max_sim.merge(master, left_on='master_side', right_index=True, how='left')

        # Update the master-series with the duplicates in cases were there is no match
        rows_to_update = dupes_max_sim[master_label].isnull()
        dupes_max_sim.loc[rows_to_update, master_label] = dupes_max_sim[rows_to_update].duplicates
        if self._master_id is not None:
            # Also update the master_id-series with the duplicates_id in cases were there is no match
            dupes_max_sim.loc[rows_to_update, master_id_label] = dupes_max_sim[rows_to_update].duplicates_id
            
            # For some weird reason, pandas' merge function changes int-datatype columns to float when NaN values
            # appear within them. So here we change them back to their original datatypes if possible:
            if dupes_max_sim[master_id_label].dtype != self._master_id.dtype and \
                self._duplicates_id.dtype == self._master_id.dtype:
                dupes_max_sim.loc[:, master_id_label] = \
                dupes_max_sim.loc[:, master_id_label].astype(self._master_id.dtype)
            
        # Prepare the output:
        required_column_list = [master_label] if self._master_id is None else [master_id_label, master_label]
        index_column_list = \
            [col for col in master.columns if col not in required_column_list] \
            if isinstance(master, pd.DataFrame) else []
        if replace_na:
            # Update the master index-columns with the duplicates index-column values in cases were there is no match
            dupes_index_columns = [col for col in dupes.columns if str(col) != 'duplicates']
            dupes_max_sim.loc[rows_to_update, index_column_list] = \
            dupes_max_sim.loc[rows_to_update, dupes_index_columns].values
            
            # Restore their original datatypes if possible:
            for m, d in zip(index_column_list, dupes_index_columns):
                if dupes_max_sim[m].dtype != master[m].dtype and dupes[d].dtype == master[m].dtype:
                    dupes_max_sim.loc[:, m] = dupes_max_sim.loc[:, m].astype(master[m].dtype)
                    
        # Make sure to keep same order as duplicates
        dupes_max_sim = dupes_max_sim.sort_values('dupe_side').set_index('dupe_side')
        output = dupes_max_sim[index_column_list + required_column_list]
        output.index = self._duplicates.index
        return output.squeeze()

    def _deduplicate(self, ignore_index=False) -> Union[pd.DataFrame, pd.Series]:
        # discard self-matches: A matches A
        pairs = self._matches_list[self._matches_list['master_side'] != self._matches_list['dupe_side']]
        # rebuild graph adjacency matrix from already found matches:
        n = len(self._master)
        graph = csr_matrix(
            (
                np.full(len(pairs), 1),
                (pairs.master_side.to_numpy(), pairs.dupe_side.to_numpy())
            ),
            shape=(n, n)
        )
        # apply scipy.csgraph's clustering algorithm (result is a 1D numpy array of length n):
        _, groups = connected_components(csgraph=graph, directed=True)
        group_of_master_index = pd.Series(groups, name='raw_group_id')

        # merge groups with string indices to obtain two-column DataFrame:
        # note: the following line automatically creates a new column named 'index' with the corresponding indices:
        group_of_master_index = group_of_master_index.reset_index()

        # Determine weights for obtaining group representatives:
        # 1. option-setting group_rep='first':
        group_of_master_index.rename(columns={'index': 'weight'}, inplace=True)
        method = 'first'
        # 2. option-setting group_rep='centroid':
        if self._config.group_rep == GROUP_REP_CENTROID:
            # reuse the adjacency matrix built above (change the 1's to corresponding cosine similarities):
            graph.data = pairs['similarity'].to_numpy()
            # sum along the rows to obtain numpy 1D matrix of similarity aggregates then ...
            # ... convert to 1D numpy array (using asarray then squeeze) and then to Series:
            group_of_master_index['weight'] = pd.Series(np.asarray(graph.sum(axis=1)).squeeze())
            method = 'idxmax'

        # Determine the group representatives AND merge with indices:
        # pandas groupby transform function and enlargement enable both respectively in one step:
        group_of_master_index['group_rep'] = \
            group_of_master_index.groupby('raw_group_id', sort=False)['weight'].transform(method)

        # Prepare the output:
        prefix = GROUP_REP_PREFIX
        label = f'{prefix}{self._master.name}' if self._master.name else prefix[:-1]
        # use group rep indexes obtained in the last step above to select the corresponding strings:
        output = self._master.iloc[group_of_master_index.group_rep].rename(label).reset_index(drop=ignore_index)
        if isinstance(output, pd.DataFrame):
            output.rename(
                columns={col: f'{prefix}{col}' for col in output.columns if str(col) != label},
                inplace=True
            )
        if self._master_id is not None:
            id_label = f'{prefix}{self._master_id.name if self._master_id.name else DEFAULT_ID_NAME}'
            # use group rep indexes obtained above to select the corresponding string IDs:
            output_id = self._master_id.iloc[group_of_master_index.group_rep].rename(id_label).reset_index(drop=True)
            output = pd.concat([output_id, output], axis=1)
        output.index = self._master.index
        return output.squeeze()

    def _get_indices_of(self, master_side: str, dupe_side: str) -> Tuple[pd.Series, pd.Series]:
        master_strings = self._master
        dupe_strings = self._master if self._duplicates is None else self._duplicates
        # Check if input is valid:
        self._validate_strings_exist(master_side, dupe_side, master_strings, dupe_strings)
        # Get the indices of the two strings
        master_indices = master_strings[master_strings == master_side].index.to_series().reset_index(drop=True)
        dupe_indices = dupe_strings[dupe_strings == dupe_side].index.to_series().reset_index(drop=True)
        return master_indices, dupe_indices
    
    def _validate_group_rep_specs(self):
        group_rep_options = (GROUP_REP_FIRST, GROUP_REP_CENTROID)
        if self._config.group_rep not in group_rep_options:
            raise Exception(
                f"Invalid option value for group_rep. The only permitted values are\n {group_rep_options}"
            )

    def _validate_replace_na_and_drop(self):
        if self._config.ignore_index and self._config.replace_na:
            raise Exception("replace_na can only be set to True when ignore_index=False.")
        if self._config.replace_na and self._master.index.nlevels != self._duplicates.index.nlevels:
            raise Exception(
                "replace_na=True: Cannot replace NaN values of index-columns with the values of another "
                "index if the number of index-levels does not equal the number of index-columns."
            )

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
        elif series_to_test.to_frame().applymap(
                    lambda x: not isinstance(x, str)
                ).squeeze().any():
            return False
        return True

    @staticmethod
    def _is_input_data_combination_valid(duplicates, master_id, duplicates_id) -> bool:
        if duplicates is None and (duplicates_id is not None) \
                or duplicates is not None and ((master_id is None) ^ (duplicates_id is None)):
            return False
        else:
            return True

    @staticmethod
    def _validate_id_data(master, duplicates, master_id, duplicates_id):
        if master_id is not None and len(master) != len(master_id):
            raise Exception('Both master and master_id must be pandas.Series of the same length.')
        if duplicates is not None and duplicates_id is not None and len(duplicates) != len(duplicates_id):
            raise Exception('Both duplicates and duplicates_id must be pandas.Series of the same length.')
