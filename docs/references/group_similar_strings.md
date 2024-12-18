---
title: group_similar_strings
---

Takes a single `Series` of strings (`strings_to_group`) and groups them by assigning to each string one string from `strings_to_group` chosen as the group-representative for each group of similar strings found. (See [tutorials/group_representatives.md](https://github.com/Bergvca/string_grouper/blob/master/tutorials/group_representatives.md) for details on how the the group-representatives are chosen.)   

If `ignore_index=True`, the output is a `Series` (with the same name as `strings_to_group` prefixed by the string `'group_rep_'`) of the same length and index as `strings_to_group` containing the group-representative strings.  If `strings_to_group` has no name then the name of the returned `Series` is `'group_rep'`.  

For example, an input Series with values: `\['foooo', 'foooob', 'bar'\]` will return `\['foooo', 'foooo', 'bar'\]`.  Here `'foooo'` and `'foooob'` are grouped together into group `'foooo'` because they are found to be similar.  Another example can be found [below](#dedup).

If `ignore_index=False`, the output is a `DataFrame` containing the above output `Series` as one of its columns with the same name.  The remaining column(s) correspond to the index (or index-levels) of `strings_to_group` and contain the index-labels of the group-representatives as values.  These columns have the same names as their counterparts prefixed by the string `'group_rep_'`. 

If `strings_id` is also given, then the IDs from `strings_id` corresponding to the group-representatives are also returned in an additional column (with the same name as `strings_id` prefixed as described above).  If `strings_id` has no name, it is assumed to have the name `'id'` before being prefixed.
   

