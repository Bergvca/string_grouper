
## `match_strings` 

   Returns a `DataFrame` containing similarity-scores of all matching pairs of highly similar strings from `master` (and `duplicates` if given).  Each matching pair in the output appears in its own row/record consisting of
   
   1. its "left" part: a string (with/without its index-label) from `master`, 
   2. its similarity score, and  
   3. its "right" part: a string (with/without its index-label) from `duplicates` (or `master` if `duplicates` is not given), 
   
   in that order.  Thus the column-names of the output are a collection of three groups:
   
   1. The name of `master` and the name(s) of its index (or index-levels) all prefixed by the string `'left_'`,
   2. `'similarity'` whose column has the similarity-scores as values, and 
   3. The name of `duplicates` (or `master` if `duplicates` is not given) and the name(s) of its index (or index-levels) prefixed by the string `'right_'`.
   
   Indexes (or their levels) only appear when the keyword argument `ignore_index=False` (the default). (See [tutorials/ignore_index_and_replace_na.md](https://github.com/Bergvca/string_grouper/blob/master/tutorials/ignore_index_and_replace_na.md) for a demonstration.)
   
   If either `master` or `duplicates` has no name, it assumes the name `'side'` which is then prefixed as described above.  Similarly, if any of the indexes (or index-levels) has no name it assumes its `pandas` default name (`'index'`, `'level_0'`, and so on) and is then prefixed as described above.
   
   In other words, if only parameter `master` is given, the function will return pairs of highly similar strings within `master`.  This can be seen as a self-join where both `'left_'` and `'right_'` prefixed columns come from `master`. If both parameters `master` and `duplicates` are given, it will return pairs of highly similar strings between `master` and `duplicates`. This can be seen as an inner-join where `'left_'` and `'right_'` prefixed columns come from `master` and `duplicates` respectively.     
   
   The function also supports optionally inputting IDs (`master_id` and `duplicates_id`) corresponding to the strings being matched.  In which case, the output includes two additional columns whose names are the names of these optional `Series` prefixed by `'left_'` and `'right_'` accordingly, and containing the IDs corresponding to the strings in the output.  If any of these `Series` has no name, then it assumes the name `'id'` and is then prefixed as described above.
   
   
## `match_most_similar` 

   If `ignore_index=True`, returns a `Series` of strings, where for each string in `duplicates` the most similar string in `master` is returned.  If there are no similar strings in `master` for a given string in `duplicates` (because there is no potential match where the cosine similarity is above the threshold \[default: 0.8\]) then the original string in `duplicates` is returned.  The output `Series` thus has the same length and index as `duplicates`.  
   
   For example, if an input `Series` with the values `\['foooo', 'bar', 'baz'\]` is passed as the argument `master`, and `\['foooob', 'bar', 'new'\]` as the values of the argument `duplicates`, the function will return a `Series` with values: `\['foooo', 'bar', 'new'\]`.
   
   The name of the output `Series` is the same as that of `master` prefixed with the string `'most_similar_'`.  If `master` has no name, it is assumed to have the name `'master'` before being prefixed.
       
   If `ignore_index=False` (the default), `match_most_similar` returns a `DataFrame` containing the same `Series` described above as one of its columns.  So it inherits the same index and length as `duplicates`.  The rest of its columns correspond to the index (or index-levels) of `master` and thus contain the index-labels of the most similar strings being output as values.  If there are no similar strings in `master` for a given string in `duplicates` then the value(s) assigned to this index-column(s) for that string is `NaN` by default.  However, if the keyword argument `replace_na=True`, then these `NaN` values are replaced with the index-label(s) of that string in `duplicates`.  Note that such replacements can only occur if the indexes of `master` and `duplicates` have the same number of levels.  (See [tutorials/ignore_index_and_replace_na.md](https://github.com/Bergvca/string_grouper/blob/master/tutorials/ignore_index_and_replace_na.md#MMS) for a demonstration.)
   
   Each column-name of the output `DataFrame` has the same name as its corresponding column, index, or index-level of `master` prefixed with the string `'most_similar_'`.
  
   If both parameters `master_id` and `duplicates_id` are also given, then a `DataFrame` is always returned with the same column(s) as described above, but with an additional column containing those IDs from these input `Series` corresponding to the output strings.  This column's name is the same as that of `master_id` prefixed in the same way as described above.  If `master_id` has no name, it is assumed to have the name `'master_id'` before being prefixed.


## `group_similar_strings` 

  Takes a single `Series` of strings (`strings_to_group`) and groups them by assigning to each string one string from `strings_to_group` chosen as the group-representative for each group of similar strings found. (See [tutorials/group_representatives.md](https://github.com/Bergvca/string_grouper/blob/master/tutorials/group_representatives.md) for details on how the the group-representatives are chosen.)   
  
  If `ignore_index=True`, the output is a `Series` (with the same name as `strings_to_group` prefixed by the string `'group_rep_'`) of the same length and index as `strings_to_group` containing the group-representative strings.  If `strings_to_group` has no name then the name of the returned `Series` is `'group_rep'`.  
   
  For example, an input Series with values: `\['foooo', 'foooob', 'bar'\]` will return `\['foooo', 'foooo', 'bar'\]`.  Here `'foooo'` and `'foooob'` are grouped together into group `'foooo'` because they are found to be similar.  Another example can be found [below](#dedup).
  
   If `ignore_index=False`, the output is a `DataFrame` containing the above output `Series` as one of its columns with the same name.  The remaining column(s) correspond to the index (or index-levels) of `strings_to_group` and contain the index-labels of the group-representatives as values.  These columns have the same names as their counterparts prefixed by the string `'group_rep_'`. 
   
   If `strings_id` is also given, then the IDs from `strings_id` corresponding to the group-representatives are also returned in an additional column (with the same name as `strings_id` prefixed as described above).  If `strings_id` has no name, it is assumed to have the name `'id'` before being prefixed.
   

## `compute_pairwise_similarities`

   Returns a `Series` of cosine similarity scores the same length and index as `string_series_1`.  Each score is the cosine similarity between the pair of strings in the same position (row) in the two input `Series`, `string_series_1` and `string_series_2`, as the position of the score in the output `Series`.  This can be seen as an element-wise comparison between the two input `Series`.
   

All functions are built using a class **`StringGrouper`**. This class can be used through pre-defined functions, for example the four high level functions above, as well as using a more interactive approach where matches can be added or removed if needed by calling the **`StringGrouper`** class directly.
   

## Options

### <a name="kwargs"></a>`kwargs`

   All keyword arguments not mentioned in the function definitions above are used to update the default settings. The following optional arguments can be used:

   * **`ngram_size`**: The amount of characters in each n-gram. Default is `3`.
   * **`regex`**: The regex string used to clean-up the input string. Default is `r"[,-./]|\s"`.
   * **`ignore_case`**: Determines whether or not letter case in strings should be ignored. Defaults to `True`.
   * **`tfidf_matrix_dtype`**: The datatype for the tf-idf values of the matrix components. Allowed values are `numpy.float32` and `numpy.float64`.  Default is `numpy.float32`.  (Note: `numpy.float32` often leads to faster processing and a smaller memory footprint albeit less numerical precision than `numpy.float64`.)
   * **`max_n_matches`**: The maximum number of matching strings in `master` allowed per string in `duplicates`. Default is the total number of strings in `master`.
   * **`min_similarity`**: The minimum cosine similarity for two strings to be considered a match.
    Defaults to `0.8`
   * **`number_of_processes`**: The number of processes used by the cosine similarity calculation. Defaults to
    `number of cores on a machine - 1.`
   * **`ignore_index`**: Determines whether indexes are ignored or not.  If `False` (the default), index-columns will appear in the output, otherwise not.  (See [tutorials/ignore_index_and_replace_na.md](https://github.com/Bergvca/string_grouper/blob/master/tutorials/ignore_index_and_replace_na.md) for a demonstration.)
   * **`replace_na`**: For function `match_most_similar`, determines whether `NaN` values in index-columns are replaced or not by index-labels from `duplicates`. Defaults to `False`.  (See [tutorials/ignore_index_and_replace_na.md](https://github.com/Bergvca/string_grouper/blob/master/tutorials/ignore_index_and_replace_na.md) for a demonstration.)
   * **`include_zeroes`**: When `min_similarity` &le; 0, determines whether zero-similarity matches appear in the output.  Defaults to `True`.  (See [tutorials/zero_similarity.md](https://github.com/Bergvca/string_grouper/blob/master/tutorials/zero_similarity.md).)  **Note:** If `include_zeroes` is `True` and the kwarg `max_n_matches` is set then it must be sufficiently high to capture ***all*** nonzero-similarity-matches, otherwise an error is raised and `string_grouper` suggests an alternative value for `max_n_matches`.  To allow `string_grouper` to automatically use the appropriate value for `max_n_matches` then do not set this kwarg at all.
   * **`group_rep`**: For function `group_similar_strings`, determines how group-representatives are chosen.  Allowed values are `'centroid'` (the default) and `'first'`.  See [tutorials/group_representatives.md](https://github.com/Bergvca/string_grouper/blob/master/tutorials/group_representatives.md) for an explanation.
   * **`force_symmetries`**: In cases where `duplicates` is `None`, specifies whether corrections should be made to the results to account for symmetry, thus compensating for those losses of numerical significance which violate the symmetries. Defaults to `True`.
   * **`n_blocks`**: This parameter is a tuple of two `int`s provided to help boost performance, if possible, of processing large DataFrames (see [Subsection Performance](#perf)), by splitting the DataFrames into `n_blocks[0]` blocks for the left operand (of the underlying matrix multiplication) and into `n_blocks[1]` blocks for the right operand before performing the string-comparisons block-wise.  Defaults to `None`, in which case automatic splitting occurs if an `OverflowError` would otherwise occur.
