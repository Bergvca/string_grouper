---
title: String Grouper
---

**`string_grouper`** is a library that makes finding groups of similar strings within a single, or multiple, lists of strings easy — and fast. **`string_grouper`** uses **tf-idf** to calculate [**cosine similarities**](http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/) within a single list or between two lists of strings. The full process is described in the blog [Super Fast String Matching in Python](https://bergvca.github.io/2017/10/14/super-fast-string-matching.html).

## Install

```bash
pip install string-grouper
```

or see the releases [here](https://github.com/bergvca/string_grouper/releases)

## First usage

```python
import pandas as pd
from string_grouper import match_strings

#https://github.com/ngshya/pfsm/blob/master/data/sec_edgar_company_info.csv
company_names = './data/sec_edgar_company_info.csv'
# We only look at the first 50k as an example:
companies = pd.read_csv(company_names)[0:50000]
# Create all matches:
matches = match_strings(companies['Company Name'])
# Look at only the non-exact matches:
matches[matches['left_Company Name'] != matches['right_Company Name']].head()
```

As shown above, the library may be used together with `pandas`, and contains four high level functions (`match_strings`, `match_most_similar`, `group_similar_strings`, and `compute_pairwise_similarities`) that can be used directly, and one class (`StringGrouper`) that allows for a more interactive approach. 

The permitted calling patterns of the four functions, and their return types, are:

| Function        | Parameters | `pandas` Return Type |
| -------------: |:-------------|:-----:|
| `match_strings`| `(master, **kwargs)`| `DataFrame` |
| `match_strings`| `(master, duplicates, **kwargs)`| `DataFrame` |
| `match_strings`| `(master, master_id=id_series, **kwargs)`| `DataFrame` |
| `match_strings`| `(master, duplicates, master_id, duplicates_id, **kwargs)`| `DataFrame` |


## With Polars

For the moment polars is not yet supported natively.

But you can juggle easily one with the other:

```python
import polars as pl
from string_grouper import match_strings

company_names = 'https://raw.githubusercontent.com/ngshya/pfsm/refs/heads/master/data/sec_edgar_company_info.csv'
# We only look at the first 50k as an example:
companies = pl.read_csv(company_names).slice(0,50000).to_pandas()
# Create all matches:
matches = pl.from_pandas(match_strings(companies['Company Name']))
# Look at only the non-exact matches:
matches.filter(pl.col('left_Company Name') != pl.col('right_Company Name')).head()
```

## High Level Functions
In the rest of this document the names, `Series` and `DataFrame`, refer to the familiar `pandas` object types.

As shown above, the library may be used together with `pandas`, and contains four high level functions (`match_strings`, `match_most_similar`, `group_similar_strings`, and `compute_pairwise_similarities`) that can be used directly, and one class (`StringGrouper`) that allows for a more interactive approach. 

The permitted calling patterns of the four functions, and their return types, are:

| Function        | Parameters | `pandas` Return Type |
| -------------: |:-------------|:-----:|
| `match_strings`| `(master, **kwargs)`| `DataFrame` |
| `match_strings`| `(master, duplicates, **kwargs)`| `DataFrame` |
| `match_strings`| `(master, master_id=id_series, **kwargs)`| `DataFrame` |
| `match_strings`| `(master, duplicates, master_id, duplicates_id, **kwargs)`| `DataFrame` |
| `match_most_similar`| `(master, duplicates, **kwargs)`| `Series` (if kwarg `ignore_index=True`) otherwise `DataFrame` (default)|
| `match_most_similar`| `(master, duplicates, master_id, duplicates_id, **kwargs)`| `DataFrame` |
| `group_similar_strings`| `(strings_to_group, **kwargs)`| `Series` (if kwarg `ignore_index=True`) otherwise `DataFrame` (default)|
| `group_similar_strings`| `(strings_to_group, strings_id, **kwargs)`| `DataFrame` |
| `compute_pairwise_similarities`| `(string_series_1, string_series_2, **kwargs)`| `Series` |



## Generic Parameters

|Name | Description |
|:--- | :--- |
|**`master`** | A `Series` of strings to be matched with themselves (or with those in `duplicates`). |
|**`duplicates`** | A `Series` of strings to be matched with those of `master`. |
|**`master_id`** (or `id_series`) | A `Series` of IDs corresponding to the strings in `master`. |
|**`duplicates_id`** | A `Series` of IDs corresponding to the strings in `duplicates`. |
|**`strings_to_group`** | A `Series` of strings to be grouped. |
|**`strings_id`** | A `Series` of IDs corresponding to the strings in `strings_to_group`. |
|**`string_series_1(_2)`** | A `Series` of strings each of which is to be compared with its corresponding string in `string_series_2(_1)`. |
|**`**kwargs`** | Keyword arguments (see [below](#kwargs)).|


## StringGrouper Class

The above-mentioned functions are all build using the [StringGrouper](references/sg_class.md) class. This class can be used for more 
each of the high-level functions listed above also has a `StringGrouper` 
method counterpart of the same name and parameters.  Calling such a method of any instance of `StringGrouper` will not 
rebuild the instance's underlying corpus to make string-comparisons but rather use it to perform the string-comparisons.  
The input Series to the method (`master`, `duplicates`, and so on) will thus be encoded, 
or transformed, into tf-idf matrices, using this corpus.  See [StringGrouper](references/sg_class.md) for further 
details. 