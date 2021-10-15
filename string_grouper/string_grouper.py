import pandas as pd
import numpy as np
import re
import multiprocessing
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from scipy.sparse.csr import csr_matrix
from scipy.sparse.lil import lil_matrix
from scipy.sparse.csgraph import connected_components
from typing import Tuple, NamedTuple, List, Optional, Union
from sparse_dot_topn_for_blocks import awesome_cossim_topn
from topn import awesome_hstack_topn
from functools import wraps


DEFAULT_NGRAM_SIZE: int = 3
DEFAULT_TFIDF_MATRIX_DTYPE: type = np.float32   # (only types np.float32 and np.float64 are allowed by sparse_dot_topn)
DEFAULT_REGEX: str = r'[,-./]|\s'
DEFAULT_MAX_N_MATCHES: int = 20
DEFAULT_MIN_SIMILARITY: float = 0.8  # minimum cosine similarity for an item to be considered a match
DEFAULT_N_PROCESSES: int = multiprocessing.cpu_count() - 1
DEFAULT_IGNORE_CASE: bool = True  # ignores case by default
DEFAULT_DROP_INDEX: bool = False  # includes index-columns in output
DEFAULT_REPLACE_NA: bool = False    # when finding the most similar strings, does not replace NaN values in most
# similar string index-columns with corresponding duplicates-index values
DEFAULT_INCLUDE_ZEROES: bool = True  # when the minimum cosine similarity <=0, determines whether zero-similarity
# matches appear in the output
GROUP_REP_CENTROID: str = 'centroid'    # Option value to select the string in each group with the largest
# similarity aggregate as group-representative:
GROUP_REP_FIRST: str = 'first'  # Option value to select the first string in each group as group-representative:
DEFAULT_GROUP_REP: str = GROUP_REP_CENTROID  # chooses group centroid as group-representative by default
DEFAULT_FORCE_SYMMETRIES: bool = True  # Option value to specify whether corrections should be made to the results
# to account for symmetry thus compensating for those numerical errors that violate symmetry due to loss of
# significance
DEFAULT_N_BLOCKS: Tuple[int, int] = None  # Option value to use to split dataset(s) into roughly equal-sized blocks

# The following string constants are used by (but aren't [yet] options passed to) StringGrouper
DEFAULT_COLUMN_NAME: str = 'side'   # used to name non-index columns of the output of StringGrouper.get_matches
DEFAULT_ID_NAME: str = 'id'  # used to name id-columns in the output of StringGrouper.get_matches
LEFT_PREFIX: str = 'left_'  # used to prefix columns on the left of the output of StringGrouper.get_matches
RIGHT_PREFIX: str = 'right_'    # used to prefix columns on the right of the output of StringGrouper.get_matches
MOST_SIMILAR_PREFIX: str = 'most_similar_'  # used to prefix columns of the output of
# StringGrouper._get_nearest_matches
DEFAULT_MASTER_NAME: str = 'master'  # used to name non-index column of the output of StringGrouper.get_nearest_matches
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
    sg = StringGrouper(string_series_1, string_series_2, **kwargs)
    return sg.dot()


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
    sg = StringGrouper(strings_to_group,
                       master_id=string_ids,
                       **kwargs)
    sg = sg.fit()
    return sg.get_groups()


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
    kwargs['max_n_matches'] = 1
    sg = StringGrouper(master,
                       duplicates=duplicates,
                       master_id=master_id,
                       duplicates_id=duplicates_id,
                       **kwargs)
    sg = sg.fit()
    return sg.get_groups()


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
    sg = StringGrouper(master,
                       duplicates=duplicates,
                       master_id=master_id,
                       duplicates_id=duplicates_id,
                       **kwargs)
    sg = sg.fit()
    return sg.get_matches()


class StringGrouperConfig(NamedTuple):
    r"""
    Class with configuration variables.

    :param ngram_size: int. The amount of characters in each n-gram. Default is 3.
    :param tfidf_matrix_dtype: type. The datatype for the tf-idf values of the matrix components.
    Possible values allowed by sparse_dot_topn are np.float32 and np.float64.  Default is np.float32.
    (Note: np.float32 often leads to faster processing and a smaller memory footprint albeit less precision
    than np.float64.)
    :param regex: str. The regex string used to cleanup the input string. Default is '[,-./]|\s'.
    :param max_n_matches: int. The maximum number of matching strings in master allowed per string in duplicates.
    Default is the total number of strings in master.
    :param min_similarity: float. The minimum cosine similarity for two strings to be considered a match.
    Defaults to 0.8.
    :param number_of_processes: int. The number of processes used by the cosine similarity calculation.
    Defaults to number of cores on a machine - 1.
    :param ignore_case: bool. Whether or not case should be ignored. Defaults to True (ignore case).
    :param ignore_index: whether or not to exclude string Series index-columns in output.  Defaults to False.
    :param include_zeroes: when the minimum cosine similarity <=0, determines whether zero-similarity matches
    appear in the output.  Defaults to True.
    :param replace_na: whether or not to replace NaN values in most similar string index-columns with
    corresponding duplicates-index values. Defaults to False.
    :param group_rep: str.  The scheme to select the group-representative.  Default is 'centroid'.
    The other choice is 'first'.
    :param force_symmetries: bool. In cases where duplicates is None, specifies whether corrections should be
    made to the results to account for symmetry, thus compensating for those losses of numerical significance
    which violate the symmetries. Defaults to True.
    :param n_blocks: (int, int) This parameter is provided to help boost performance, if possible, of
    processing large DataFrames, by splitting the DataFrames into n_blocks[0] blocks for the left
    operand (of the underlying matrix multiplication) and into n_blocks[1] blocks for the right operand
    before performing the string-comparisons block-wise.  Defaults to None.
    """

    ngram_size: int = DEFAULT_NGRAM_SIZE
    tfidf_matrix_dtype: int = DEFAULT_TFIDF_MATRIX_DTYPE
    regex: str = DEFAULT_REGEX
    max_n_matches: Optional[int] = None
    min_similarity: float = DEFAULT_MIN_SIMILARITY
    number_of_processes: int = DEFAULT_N_PROCESSES
    ignore_case: bool = DEFAULT_IGNORE_CASE
    ignore_index: bool = DEFAULT_DROP_INDEX
    include_zeroes: bool = DEFAULT_INCLUDE_ZEROES
    replace_na: bool = DEFAULT_REPLACE_NA
    group_rep: str = DEFAULT_GROUP_REP
    force_symmetries: bool = DEFAULT_FORCE_SYMMETRIES
    n_blocks: Tuple[int, int] = DEFAULT_N_BLOCKS


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
        # private members:
        self.is_build = False

        self._master: pd.DataFrame = pd.DataFrame()
        self._duplicates: Optional[pd.Series] = None
        self._master_id: Optional[pd.Series] = None
        self._duplicates_id: Optional[pd.Series] = None

        self._right_Series: pd.DataFrame = pd.DataFrame()
        self._left_Series: pd.DataFrame = pd.DataFrame()

        # After the StringGrouper is fit, _matches_list will contain the indices and similarities of the matches
        self._matches_list: pd.DataFrame = pd.DataFrame()
        # _true_max_n_matches will contain the true maximum number of matches over all strings in master if
        # self._config.min_similarity <= 0
        self._true_max_n_matches: int = 0
        self._max_n_matches: int = 0

        self._config: StringGrouperConfig = StringGrouperConfig(**kwargs)

        # initialize the members:
        self._set_data(master, duplicates, master_id, duplicates_id)
        self._set_options(**kwargs)
        self._build_corpus()

    def _set_data(self,
                  master: pd.Series,
                  duplicates: Optional[pd.Series] = None,
                  master_id: Optional[pd.Series] = None,
                  duplicates_id: Optional[pd.Series] = None):
        # Validate input strings data
        self.master = master
        self.duplicates = duplicates

        # Validate optional IDs input
        if not StringGrouper._is_input_data_combination_valid(duplicates, master_id, duplicates_id):
            raise Exception('List of data Series options is invalid')
        StringGrouper._validate_id_data(master, duplicates, master_id, duplicates_id)
        self._master_id = master_id
        self._duplicates_id = duplicates_id

        # Set some private members
        self._right_Series = self._master
        if self._duplicates is None:
            self._left_Series = self._master
        else:
            self._left_Series = self._duplicates

        self.is_build = False

    def _set_options(self, **kwargs):
        self._config = StringGrouperConfig(**kwargs)

        if self._config.max_n_matches is None:
            self._max_n_matches = len(self._master)
        else:
            self._max_n_matches = self._config.max_n_matches

        self._validate_group_rep_specs()
        self._validate_tfidf_matrix_dtype()
        self._validate_replace_na_and_drop()
        StringGrouper._validate_n_blocks(self._config.n_blocks)
        self.is_build = False

    def _build_corpus(self):
        self._vectorizer = TfidfVectorizer(min_df=1, analyzer=self.n_grams, dtype=self._config.tfidf_matrix_dtype)
        self._vectorizer = self._fit_vectorizer()
        self.is_build = False  # indicates if the grouper was fit or not

    def reset_data(self,
                   master: pd.Series,
                   duplicates: Optional[pd.Series] = None,
                   master_id: Optional[pd.Series] = None,
                   duplicates_id: Optional[pd.Series] = None):
        """
        Sets the input Series of a StringGrouper instance without changing the underlying corpus.
        :param master: pandas.Series. A Series of strings in which similar strings are searched, either against itself
        or against the `duplicates` Series.
        :param duplicates: pandas.Series. If set, for each string in duplicates a similar string is searched in Master.
        :param master_id: pandas.Series. If set, contains ID values for each row in master Series.
        :param duplicates_id: pandas.Series. If set, contains ID values for each row in duplicates Series.
        :param kwargs: All other keyword arguments are passed to StringGrouperConfig
        """
        self._set_data(master, duplicates, master_id, duplicates_id)

    def clear_data(self):
        self._master = None
        self._duplicates = None
        self._master_id = None
        self._duplicates_id = None
        self._matches_list = None
        self._left_Series = None
        self._right_Series = None
        self.is_build = False

    def update_options(self, **kwargs):
        """
        Updates the kwargs of a StringGrouper object
        :param **kwargs: any StringGrouper keyword=value argument pairs
        """
        _ = StringGrouperConfig(**kwargs)
        old_kwargs = self._config._asdict()
        old_kwargs.update(kwargs)
        self._set_options(**old_kwargs)

    @property
    def master(self):
        return self._master

    @master.setter
    def master(self, master):
        if not StringGrouper._is_series_of_strings(master):
            raise TypeError('Master input does not consist of pandas.Series containing only Strings')
        self._master = master

    @property
    def duplicates(self):
        return self._duplicates

    @duplicates.setter
    def duplicates(self, duplicates):
        if duplicates is not None and not StringGrouper._is_series_of_strings(duplicates):
            raise TypeError('Duplicates input does not consist of pandas.Series containing only Strings')
        self._duplicates = duplicates

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

    def _fit_blockwise_manual(self, n_blocks=(1, 1)):
        # Function to compute matrix product by optionally first dividing
        # the DataFrames(s) into equal-sized blocks as much as possible.

        def divide_by(n, series):
            # Returns an array of n rows and 2 columns.
            # The columns denote the start and end of each of the n blocks.
            # Note: zero-indexing is implied.
            sz = len(series)//n
            block_rem = np.full(n, 0, dtype=np.int64)
            block_rem[:len(series) % n] = 1
            if sz > 0:
                equal_block_sz = np.full(n, sz, dtype=np.int64)
                equal_block_sz += block_rem
            else:
                equal_block_sz = block_rem[:len(series) % n]
            equal_block_sz = np.cumsum(equal_block_sz)
            equal_block_sz = np.tile(equal_block_sz, (2, 1))
            equal_block_sz[0, 0] = 0
            equal_block_sz[0, 1:] = equal_block_sz[1, :-1]
            return equal_block_sz.T

        block_ranges_left = divide_by(n_blocks[0], self._left_Series)
        block_ranges_right = divide_by(n_blocks[1], self._right_Series)

        self._true_max_n_matches = 0
        block_true_max_n_matches = 0
        vblocks = []
        for left_block in block_ranges_left:
            left_matrix = self._get_left_tf_idf_matrix(left_block)
            nnz_rows = np.full(left_matrix.shape[0], 0, dtype=np.int32)
            hblocks = []
            for right_block in block_ranges_right:
                right_matrix = self._get_right_tf_idf_matrix(right_block)
                try:
                    # Calculate the matches using the cosine similarity
                    # Note: awesome_cossim_topn will sort each row only when
                    # _max_n_matches < size of right_block or sort=True
                    matches, block_true_max_n_matches = self._build_matches(
                        left_matrix, right_matrix, nnz_rows, sort=(len(block_ranges_right) == 1)
                    )
                except OverflowError as oe:
                    import sys
                    raise (type(oe)(f"{str(oe)} Use the n_blocks parameter to split-up "
                                    f"the data into smaller chunks.  The current values"
                                    f"(n_blocks = {n_blocks}) are too small.")
                           .with_traceback(sys.exc_info()[2]))
                hblocks.append(matches)
                # end of inner loop

            self._true_max_n_matches = \
                max(block_true_max_n_matches, self._true_max_n_matches)
            if len(block_ranges_right) > 1:
                # Note: awesome_hstack_topn will sort each row only when
                # _max_n_matches < length of _right_Series or sort=True
                vblocks.append(
                    awesome_hstack_topn(
                        hblocks,
                        self._max_n_matches,
                        sort=True,
                        use_threads=self._config.number_of_processes > 1,
                        n_jobs=self._config.number_of_processes
                    )
                )
            else:
                vblocks.append(hblocks[0])
            del hblocks
            del matches
            # end of outer loop

        if len(block_ranges_left) > 1:
            return vstack(vblocks)
        else:
            return vblocks[0]

    def _fit_blockwise_auto(self,
                            left_partition=(None, None),
                            right_partition=(None, None),
                            nnz_rows=None,
                            sort=True,
                            whoami=0):
        # This is a recursive function!
        # fit() has been extended here to enable StringGrouper to handle large
        # datasets which otherwise would lead to an OverflowError
        # The handling is achieved using block matrix multiplication.
        def begin(partition):
            return partition[0] if partition[0] is not None else 0

        def end(partition, left=True):
            if partition[1] is not None:
                return partition[1]

            return len(self._left_Series if left else self._right_Series)

        left_matrix = self._get_left_tf_idf_matrix(left_partition)
        right_matrix = self._get_right_tf_idf_matrix(right_partition)

        if whoami == 0:
            # At the topmost level of recursion initialize nnz_rows
            # which will be used to compute _true_max_n_matches
            nnz_rows = np.full(left_matrix.shape[0], 0, dtype=np.int32)
            self._true_max_n_matches = 0

        try:
            # Calculate the matches using the cosine similarity
            matches, true_max_n_matches = self._build_matches(
                left_matrix, right_matrix, nnz_rows[slice(*left_partition)],
                sort=sort)
        except OverflowError:
            warnings.warn("An OverflowError occurred but is being "
                          "handled.  The input data will be automatically "
                          "split-up into smaller chunks which will then be "
                          "processed one chunk at a time.  To prevent "
                          "OverflowError, use the n_blocks parameter to split-up "
                          "the data manually into small enough chunks.")
            # Matrices too big!  Try splitting:
            del left_matrix, right_matrix

            def split_partition(partition, left=True):
                data_begin = begin(partition)
                data_end = end(partition, left=left)
                data_mid = data_begin + (data_end - data_begin)//2
                if data_mid > data_begin:
                    return [(data_begin, data_mid), (data_mid, data_end)]
                else:
                    return [(data_begin, data_end)]

            left_halves = split_partition(left_partition, left=True)
            right_halves = split_partition(right_partition, left=False)
            vblocks = []
            for lhalf in left_halves:
                hblocks = []
                for rhalf in right_halves:
                    # Note: awesome_cossim_topn will sort each row only when
                    # _max_n_matches < size of right_partition or sort=True
                    matches = self._fit_blockwise_auto(
                        left_partition=lhalf, right_partition=rhalf,
                        nnz_rows=nnz_rows,
                        sort=((whoami == 0) and (len(right_halves) == 1)),
                        whoami=(whoami + 1)
                    )
                    hblocks.append(matches)
                    # end of inner loop
                if whoami == 0:
                    self._true_max_n_matches = max(
                        np.amax(nnz_rows[slice(*lhalf)]),
                        self._true_max_n_matches
                    )
                if len(right_halves) > 1:
                    # Note: awesome_hstack_topn will sort each row only when
                    # _max_n_matches < length of _right_Series or sort=True
                    vblocks.append(
                        awesome_hstack_topn(
                            hblocks,
                            self._max_n_matches,
                            sort=(whoami == 0),
                            use_threads=self._config.number_of_processes > 1,
                            n_jobs=self._config.number_of_processes
                        )
                    )
                else:
                    vblocks.append(hblocks[0])
                del hblocks
                # end of outer loop
            if len(left_halves) > 1:
                return vstack(vblocks)
            else:
                return vblocks[0]

        if whoami == 0:
            self._true_max_n_matches = true_max_n_matches
        return matches

    def fit(self, force_symmetries=None, n_blocks=None):
        """
        Builds the _matches list which contains string-matches' indices and similarity
        Updates and returns the StringGrouper object that called it.
        """
        if force_symmetries is None:
            force_symmetries = self._config.force_symmetries
        StringGrouper._validate_n_blocks(n_blocks)
        if n_blocks is None:
            n_blocks = self._config.n_blocks

        # do the matching
        if n_blocks is None:
            matches = self._fit_blockwise_auto()
        else:
            matches = self._fit_blockwise_manual(n_blocks=n_blocks)

        # enforce symmetries?
        if force_symmetries and (self._duplicates is None):
            # convert to lil format for best efficiency when setting
            # matrix-elements
            matches = matches.tolil()
            # matrix diagonal elements must be exactly 1 (numerical precision
            # errors introduced by floating-point computations in
            # awesome_cossim_topn sometimes lead to unexpected results)
            matches = StringGrouper._fix_diagonal(matches)
            # the list of matches must be symmetric!
            # (i.e., if A != B and A matches B; then B matches A)
            matches = StringGrouper._symmetrize_matrix(matches)
            matches = matches.tocsr()
        self._matches_list = self._get_matches_list(matches)
        self.is_build = True
        return self

    def dot(self) -> pd.Series:
        """Computes the row-wise similarity scores between strings in _master and _duplicates"""
        if len(self._master) != len(self._duplicates):
            raise Exception("To perform this function, both input Series must have the same length.")
        master_matrix, duplicate_matrix = self._get_left_tf_idf_matrix(), self._get_right_tf_idf_matrix()
        # Calculate pairwise cosine similarities:
        pairwise_similarities = np.asarray(master_matrix.multiply(duplicate_matrix).sum(axis=1)).squeeze(axis=1)
        return pd.Series(pairwise_similarities, name='similarity', index=self._master.index)

    @validate_is_fit
    def get_matches(self,
                    ignore_index: Optional[bool] = None,
                    include_zeroes: Optional[bool] = None) -> pd.DataFrame:
        """
        Returns a DataFrame with all the matches and their cosine similarity.
        If optional IDs are used, returned as extra columns with IDs matched to respective data rows

        :param ignore_index: whether or not to exclude string Series index-columns in output.  Defaults to
        self._config.ignore_index.
        :param include_zeroes: when the minimum cosine similarity <=0, determines whether zero-similarity matches
        appear in the output.  Defaults to self._config.include_zeroes.
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

        if ignore_index is None:
            ignore_index = self._config.ignore_index
        if include_zeroes is None:
            include_zeroes = self._config.include_zeroes
        if self._config.min_similarity > 0 or not include_zeroes:
            matches_list = self._matches_list
        elif include_zeroes:
            # Here's a fix to a bug pointed out by one GitHub user (@nbcvijanovic):
            # the fix includes zero-similarity matches that are missing by default
            # in _matches_list due to our use of sparse matrices
            non_matches_list = self._get_non_matches_list()
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
        if ignore_index is None:
            ignore_index = self._config.ignore_index
        if self._duplicates is None:
            return self._deduplicate(ignore_index=ignore_index)
        else:
            if replace_na is None:
                replace_na = self._config.replace_na
            return self._get_nearest_matches(ignore_index=ignore_index, replace_na=replace_na)

    def match_strings(self,
                      master: pd.Series,
                      duplicates: Optional[pd.Series] = None,
                      master_id: Optional[pd.Series] = None,
                      duplicates_id: Optional[pd.Series] = None,
                      **kwargs) -> pd.DataFrame:
        """
        Returns all highly similar strings without rebuilding the corpus.
        If only 'master' is given, it will return highly similar strings within master.
        This can be seen as an self-join. If both master and duplicates is given, it will return highly similar strings
        between master and duplicates. This can be seen as an inner-join.

        :param master: pandas.Series. Series of strings against which matches are calculated.
        :param duplicates: pandas.Series. Series of strings that will be matched with master if given (Optional).
        :param master_id: pandas.Series. Series of values that are IDs for master column rows (Optional).
        :param duplicates_id: pandas.Series. Series of values that are IDs for duplicates column rows (Optional).
        :param kwargs: All other keyword arguments are passed to StringGrouperConfig.
        :return: pandas.Dataframe.
        """
        self.reset_data(master, duplicates, master_id, duplicates_id)
        self.update_options(**kwargs)
        self = self.fit()
        return self.get_matches()

    def match_most_similar(self,
                           master: pd.Series,
                           duplicates: pd.Series,
                           master_id: Optional[pd.Series] = None,
                           duplicates_id: Optional[pd.Series] = None,
                           **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """
        If no IDs ('master_id' and 'duplicates_id') are given, returns, without rebuilding the corpus, a
        Series of strings of the same length as 'duplicates' where for each string in duplicates the most
        similar string in 'master' is returned.
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
        self.reset_data(master, duplicates, master_id, duplicates_id)

        old_max_n_matches = self._max_n_matches
        new_max_n_matches = None
        if 'max_n_matches' in kwargs:
            new_max_n_matches = kwargs['max_n_matches']
        kwargs['max_n_matches'] = 1
        self.update_options(**kwargs)

        self = self.fit()
        output = self.get_groups()

        kwargs['max_n_matches'] = old_max_n_matches if new_max_n_matches is None else new_max_n_matches
        self.update_options(**kwargs)
        return output

    def group_similar_strings(self,
                              strings_to_group: pd.Series,
                              string_ids: Optional[pd.Series] = None,
                              **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """
        If 'string_ids' is not given, finds all similar strings in 'strings_to_group' without rebuilding the
        corpus and returns a Series of strings of the same length as 'strings_to_group'. For each group of
        similar strings a single string is chosen as the 'master' string and is returned for each member of
        the group.

        For example the input Series: [foooo, foooob, bar] will return [foooo, foooo, bar].  Here 'foooo' and
        'foooob' are grouped together into group 'foooo' because they are found to be very similar.

        If string_ids is also given, a DataFrame of the strings and their corresponding IDs is instead returned.

        :param strings_to_group: pandas.Series. The input Series of strings to be grouped.
        :param string_ids: pandas.Series. The input Series of the IDs of the strings to be grouped. (Optional)
        :param kwargs: All other keyword arguments are passed to StringGrouperConfig. (Optional)
        :return: pandas.Series or pandas.DataFrame.
        """
        self.reset_data(strings_to_group, master_id=string_ids)
        self.update_options(**kwargs)
        self = self.fit()
        return self.get_groups()

    def compute_pairwise_similarities(self,
                                      string_series_1: pd.Series,
                                      string_series_2: pd.Series,
                                      **kwargs) -> pd.Series:
        """
        Computes the similarity scores between two Series of strings row-wise without rebuilding the corpus.

        :param string_series_1: pandas.Series. The input Series of strings to be grouped
        :param string_series_2: pandas.Series. The input Series of the IDs of the strings to be grouped
        :param kwargs: All other keyword arguments are passed to StringGrouperConfig
        :return: pandas.Series of similarity scores, the same length as string_series_1 and string_series_2
        """
        self.reset_data(string_series_1, string_series_2)
        self.update_options(**kwargs)
        return self.dot()

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

    def _get_left_tf_idf_matrix(self, partition=(None, None)):
        # unlike _get_tf_idf_matrices(), _get_left_tf_idf_matrix
        # does not set the corpus but rather
        # builds a matrix using the existing corpus
        return self._vectorizer.transform(
            self._left_Series.iloc[slice(*partition)])

    def _get_right_tf_idf_matrix(self, partition=(None, None)):
        # unlike _get_tf_idf_matrices(), _get_right_tf_idf_matrix
        # does not set the corpus but rather
        # builds a matrix using the existing corpus
        return self._vectorizer.transform(
            self._right_Series.iloc[slice(*partition)])

    def _fit_vectorizer(self) -> TfidfVectorizer:
        # if both dupes and master string series are set - we concat them to fit the vectorizer on all
        # strings
        if self._duplicates is not None:
            strings = pd.concat([self._master, self._duplicates])
        else:
            strings = self._master
        self._vectorizer.fit(strings)
        return self._vectorizer

    def _build_matches(self,
                       left_matrix: csr_matrix, right_matrix: csr_matrix,
                       nnz_rows: np.ndarray = None,
                       sort: bool = True) -> csr_matrix:
        """Builds the cossine similarity matrix of two csr matrices"""
        right_matrix = right_matrix.transpose()

        if nnz_rows is None:
            nnz_rows = np.full(left_matrix.shape[0], 0, dtype=np.int32)

        optional_kwargs = {
            'return_best_ntop': True,
            'sort': sort,
            'use_threads': self._config.number_of_processes > 1,
            'n_jobs': self._config.number_of_processes}

        return awesome_cossim_topn(
            left_matrix, right_matrix,
            self._max_n_matches,
            nnz_rows,
            self._config.min_similarity,
            **optional_kwargs)

    def _get_matches_list(self,
                          matches: csr_matrix
                          ) -> pd.DataFrame:
        """Returns a list of all the indices of matches"""
        r, c = matches.nonzero()
        d = matches.data
        return pd.DataFrame({'master_side': c.astype(np.int64),
                             'dupe_side': r.astype(np.int64),
                             'similarity': d})

    def _get_non_matches_list(self) -> pd.DataFrame:
        """Returns a list of all the indices of non-matching pairs (with similarity set to 0)"""
        m_sz, d_sz = len(self._master), len(self._master if self._duplicates is None else self._duplicates)
        all_pairs = pd.MultiIndex.from_product([range(m_sz), range(d_sz)], names=['master_side', 'dupe_side'])
        matched_pairs = pd.MultiIndex.from_frame(self._matches_list[['master_side', 'dupe_side']])
        missing_pairs = all_pairs.difference(matched_pairs)
        if missing_pairs.empty:
            return pd.DataFrame()
        if (self._max_n_matches < self._true_max_n_matches):
            raise Exception(f'\nERROR: Cannot return zero-similarity matches since \n'
                            f'\t\t max_n_matches={self._max_n_matches} is too small!\n'
                            f'\t\t Try setting max_n_matches={self._true_max_n_matches} (the \n'
                            f'\t\t true maximum number of matches over all strings in master)\n'
                            f'\t\t or greater or do not set this kwarg at all.')
        missing_pairs = missing_pairs.to_frame(index=False)
        missing_pairs['similarity'] = 0
        return missing_pairs

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
        return output.squeeze(axis=1)

    def _deduplicate(self, ignore_index=False) -> Union[pd.DataFrame, pd.Series]:
        pairs = self._matches_list
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
            group_of_master_index['weight'] = pd.Series(np.asarray(graph.sum(axis=1)).squeeze(axis=1))
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
        return output

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

    def _validate_tfidf_matrix_dtype(self):
        dtype_options = (np.float32, np.float64)
        if self._config.tfidf_matrix_dtype not in dtype_options:
            raise Exception(
                f"Invalid option value for tfidf_matrix_dtype. The only permitted values are\n {dtype_options}"
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
    def _validate_n_blocks(n_blocks):
        errmsg = "Invalid option value for parameter n_blocks: "
        "n_blocks must be None or a tuple of 2 integers greater than 0."
        if n_blocks is None:
            return
        if not isinstance(n_blocks, tuple):
            raise Exception(errmsg)
        if len(n_blocks) != 2:
            raise Exception(errmsg)
        if not (isinstance(n_blocks[0], int) and isinstance(n_blocks[1], int)):
            raise Exception(errmsg)
        if (n_blocks[0] < 1) or (n_blocks[1] < 1):
            raise Exception(errmsg)

    @staticmethod
    def _fix_diagonal(m: lil_matrix) -> lil_matrix:
        r = np.arange(m.shape[0])
        m[r, r] = 1
        return m

    @staticmethod
    def _symmetrize_matrix(m_symmetric: lil_matrix) -> lil_matrix:
        r, c = m_symmetric.nonzero()
        m_symmetric[c, r] = m_symmetric[r, c]
        return m_symmetric

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
                ).squeeze(axis=1).any():
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
