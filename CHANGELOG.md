# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.7.0] - 2025-01-23

### Changed

* sparse_dot_topn_for_blocks and topn dependencies are removed and replaced by sparse_dot_topn official library from ING Bank, this is a big change: it may have impacts from old code using string_grouper 
* `n_blocks` None is now the default value for `n_blocks` and optimal numbers of blocks will be guessed based on empirical observation to split data into smaller chunks (based on input data size)
* sparse_dot_topn now integrates a [block/chunk strategy](https://github.com/ing-bank/sparse_dot_topn?tab=readme-ov-file#distributing-the-top-n-multiplication-of-two-large-o10m-sparse-matrices-over-a-cluster). This strategy is used in string_grouper. 


### Added

* a new parameter normalize_to_ascii to normalize unicode character to ascii ones 
* loguru dependency is introduced to print messages to user


## [0.6.1] - 2021-10-19

* `n_blocks` Added "guesstimate" as default value for `n_blocks`. This will guess an optimal number of blocks
based on empirical observation.


## [0.6.0] - 2021-09-21

### Added

* matrix-blocking/splitting as a performance-enhancer (see [README.md](https://github.com/Bergvca/string_grouper/tree/master/#performance) for details)
* new keyword arguments `force_symmetries` and `n_blocks` (see [README.md](https://github.com/Bergvca/string_grouper/tree/master/#kwargs) for details)
* new dependency on packages `topn` and `sparse_dot_topn_for_blocks` to help with the matrix-blocking
* capability to reuse a previously initialized StringGrouper (that is, the corpus can now persist across high-level function calls like `match_strings()`.  See [README.md](https://github.com/Bergvca/string_grouper/tree/master/#corpus) for details.)


## [0.5.0] - 2021-06-11

### Added

* Added new keyword argument **`tfidf_matrix_dtype`** (the datatype for the tf-idf values of the matrix components). Allowed values are `numpy.float32` and `numpy.float64` (used by the required external package `sparse_dot_topn` version 0.3.1).  Default is `numpy.float32`.  (Note: `numpy.float32` often leads to faster processing and a smaller memory footprint albeit less numerical precision than `numpy.float64`.)

### Changed

* Changed dependency on `sparse_dot_topn` from version 0.2.9 to 0.3.1
* Changed the default datatype for cosine similarities from numpy.float64 to numpy.float32 to boost computational performance at the expense of numerical precision.
* Changed the default value of the keyword argument `max_n_matches` from 20 to the number of strings in `duplicates` (or `master`, if `duplicates` is not given). 
* Changed warning issued when the condition \[`include_zeroes=True` and `min_similarity` &le; 0 and `max_n_matches` is not sufficiently high to capture all nonzero-similarity-matches\] is met to an exception. 
 
### Removed

* Removed the keyword argument `suppress_warning`

## [0.4.0] - 2021-04-11

### Added

* Added group representative functionality - by default the centroid is used. From [@ParticularMiner](https://github.com/ParticularMiner)
* Added string_grouper_utils package with additional group-representative functionality: 
    * new_group_rep_by_earliest_timestamp
    * new_group_rep_by_completeness
    * new_group_rep_by_highest_weight

    From [@ParticularMiner](https://github.com/ParticularMiner)    
* Original indices are now added by default to output of `group_similar_strings`, `match_most_similar` and `match_strings`.
  From [@ParticularMiner](https://github.com/ParticularMiner)
* `compute_pairwise_similarities` function From [@ParticularMiner](https://github.com/ParticularMiner) 

### Changed

* Default group representative is now the centroid. Used to be the first string in the series belonging to a group.
  From [@ParticularMiner](https://github.com/ParticularMiner)
* Output of `match_most_similar` and `match_strings` is now a `pandas.DataFrame` object instead of a `pandas.Series`
by default. From [@ParticularMiner](https://github.com/ParticularMiner)
* Fixed a bug which occurs when min_similarity=0. From [@ParticularMiner](https://github.com/ParticularMiner)