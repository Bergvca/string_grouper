# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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