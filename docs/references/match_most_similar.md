---
title: match_most_similar
---


If `ignore_index=True`, returns a `Series` of strings, where for each string in `duplicates` the most similar string in `master` is returned.  If there are no similar strings in `master` for a given string in `duplicates` (because there is no potential match where the cosine similarity is above the threshold \[default: 0.8\]) then the original string in `duplicates` is returned.  The output `Series` thus has the same length and index as `duplicates`.  

For example, if an input `Series` with the values `\['foooo', 'bar', 'baz'\]` is passed as the argument `master`, and `\['foooob', 'bar', 'new'\]` as the values of the argument `duplicates`, the function will return a `Series` with values: `\['foooo', 'bar', 'new'\]`.

The name of the output `Series` is the same as that of `master` prefixed with the string `'most_similar_'`.  If `master` has no name, it is assumed to have the name `'master'` before being prefixed.
    
If `ignore_index=False` (the default), `match_most_similar` returns a `DataFrame` containing the same `Series` described above as one of its columns.  So it inherits the same index and length as `duplicates`.  The rest of its columns correspond to the index (or index-levels) of `master` and thus contain the index-labels of the most similar strings being output as values.  If there are no similar strings in `master` for a given string in `duplicates` then the value(s) assigned to this index-column(s) for that string is `NaN` by default.  However, if the keyword argument `replace_na=True`, then these `NaN` values are replaced with the index-label(s) of that string in `duplicates`.  Note that such replacements can only occur if the indexes of `master` and `duplicates` have the same number of levels.  (See [tutorials/ignore_index_and_replace_na.md](https://github.com/Bergvca/string_grouper/blob/master/tutorials/ignore_index_and_replace_na.md#MMS) for a demonstration.)

Each column-name of the output `DataFrame` has the same name as its corresponding column, index, or index-level of `master` prefixed with the string `'most_similar_'`.

If both parameters `master_id` and `duplicates_id` are also given, then a `DataFrame` is always returned with the same column(s) as described above, but with an additional column containing those IDs from these input `Series` corresponding to the output strings.  This column's name is the same as that of `master_id` prefixed in the same way as described above.  If `master_id` has no name, it is assumed to have the name `'master_id'` before being prefixed.

