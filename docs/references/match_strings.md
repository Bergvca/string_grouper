---
title: match_strings
---


Returns a `DataFrame` containing similarity-scores of all matching pairs of highly similar strings from `master` (and `duplicates` if given).  Each matching pair in the output appears in its own row/record consisting of
   
- 1. its "left" part: a string (with/without its index-label) from `master`, 
- 2. its similarity score, and  
- 3. its "right" part: a string (with/without its index-label) from `duplicates` (or `master` if `duplicates` is not given), 
   
in that order.  Thus the column-names of the output are a collection of three groups:
   
- 1. The name of `master` and the name(s) of its index (or index-levels) all prefixed by the - string `'left_'`,
- 2. `'similarity'` whose column has the similarity-scores as values, and 
- 3. The name of `duplicates` (or `master` if `duplicates` is not given) and the name(s) of its index (or index-levels) prefixed by the string `'right_'`.

Indexes (or their levels) only appear when the keyword argument `ignore_index=False` (the default). (See [tutorials/ignore_index_and_replace_na.md](https://github.com/Bergvca/string_grouper/blob/master/tutorials/ignore_index_and_replace_na.md) for a demonstration.)

If either `master` or `duplicates` has no name, it assumes the name `'side'` which is then prefixed as described above.  Similarly, if any of the indexes (or index-levels) has no name it assumes its `pandas` default name (`'index'`, `'level_0'`, and so on) and is then prefixed as described above.

In other words, if only parameter `master` is given, the function will return pairs of highly similar strings within `master`.  This can be seen as a self-join where both `'left_'` and `'right_'` prefixed columns come from `master`. If both parameters `master` and `duplicates` are given, it will return pairs of highly similar strings between `master` and `duplicates`. This can be seen as an inner-join where `'left_'` and `'right_'` prefixed columns come from `master` and `duplicates` respectively.     

The function also supports optionally inputting IDs (`master_id` and `duplicates_id`) corresponding to the strings being matched.  In which case, the output includes two additional columns whose names are the names of these optional `Series` prefixed by `'left_'` and `'right_'` accordingly, and containing the IDs corresponding to the strings in the output.  If any of these `Series` has no name, then it assumes the name `'id'` and is then prefixed as described above.


