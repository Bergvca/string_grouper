---
title: Options / **kwargs
---

All keyword arguments not mentioned in the function definitions above are used to update the default settings. The following optional arguments can be used:

## Tokenization settings

* **`ngram_size`**: The amount of characters in each n-gram. Default is `3`.
* **`regex`**: The regex string used to clean-up the input string. Default is `r"[,-./]|\s"`.
* **`ignore_case`**: Determines whether or not letter case in strings should be ignored. Defaults to `True`.
* **`normalize_to_ascii`**: Determines whether or not unicode to ascii normarlization is done. Default to `True`.

## Match and output settings

* **`max_n_matches`**: The maximum number of matching strings in `master` allowed per string in `duplicates`. Default is 20.
* **`min_similarity`**: The minimum cosine similarity for two strings to be considered a match.
 Defaults to `0.8`
* **`include_zeroes`**: When `min_similarity` &le; 0, determines whether zero-similarity matches appear in the output.  Defaults to `True`.  (See [tutorials/zero_similarity.md](https://github.com/Bergvca/string_grouper/blob/master/tutorials/zero_similarity.md).) 
* **`ignore_index`**: Determines whether indexes are ignored or not.  If `False` (the default), index-columns will appear in the output, otherwise not.  (See [tutorials/ignore_index_and_replace_na.md](https://github.com/Bergvca/string_grouper/blob/master/tutorials/ignore_index_and_replace_na.md) for a demonstration.)
* **`replace_na`**: For function `match_most_similar`, determines whether `NaN` values in index-columns are replaced or not by index-labels from `duplicates`. Defaults to `False`.  (See [tutorials/ignore_index_and_replace_na.md](https://github.com/Bergvca/string_grouper/blob/master/tutorials/ignore_index_and_replace_na.md) for a demonstration.)

## Performance settings

* **`number_of_processes`**: The number of processes used by the cosine similarity calculation. Defaults to
 `number of cores on a machine - 1.`
* **`n_blocks`**: This parameter is a tuple of two `int`s provided to help boost performance, if possible, of processing large DataFrames (see [Subsection Performance](#perf)), by splitting the DataFrames into `n_blocks[0]` blocks for the left operand (of the underlying matrix multiplication) and into `n_blocks[1]` blocks for the right operand before performing the string-comparisons block-wise.  Defaults to `None`, in which case automatic splitting occurs if an `OverflowError` would otherwise occur.

## Other settings

* **`tfidf_matrix_dtype`**: The datatype for the tf-idf values of the matrix components. Allowed values are `numpy.float32` and `numpy.float64`.  Default is `numpy.float64`.  (Note: `numpy.float32` often leads to faster processing and a smaller memory footprint albeit less numerical precision than `numpy.float64`.)
* **`group_rep`**: For function `group_similar_strings`, determines how group-representatives are chosen.  Allowed values are `'centroid'` (the default) and `'first'`.  See [tutorials/group_representatives.md](https://github.com/Bergvca/string_grouper/blob/master/tutorials/group_representatives.md) for an explanation.
