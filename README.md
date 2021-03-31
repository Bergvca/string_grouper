# String Grouper  
<!-- Some cool decorations -->
[![pypi](https://badgen.net/pypi/v/string-grouper)](https://pypi.org/project/string-grouper)
[![license](https://badgen.net/pypi/license/string_grouper)](https://github.com/Bergvca/string_grouper)
[![lastcommit](https://badgen.net/github/last-commit/Bergvca/string_grouper)](https://github.com/Bergvca/string_grouper)
[![codecov](https://codecov.io/gh/Bergvca/string_grouper/branch/master/graph/badge.svg?token=AGK441CQDT)](https://codecov.io/gh/Bergvca/string_grouper)
<!-- [![github](https://shields.io/github/v/release/Bergvca/string_grouper)](https://github.com/Bergvca/string_grouper) -->

**<samp>string_grouper</samp>** is a library that makes finding groups of similar strings within a single, or multiple, lists of strings easy — and fast. **<samp>string_grouper</samp>** uses **tf-idf** to calculate [**cosine similarities**](https://towardsdatascience.com/understanding-cosine-similarity-and-its-application-fd42f585296a) within a single list or between two lists of strings. The full process is described in the blog [Super Fast String Matching in Python](https://bergvca.github.io/2017/10/14/super-fast-string-matching.html).

## Installing

<samp>pip install string-grouper</samp>

## Usage

```python
import pandas as pd
from string_grouper import match_strings, match_most_similar, \
	group_similar_strings, compute_pairwise_similarities, \
	StringGrouper
```

As shown above, the library may be used together with <samp>pandas</samp>, and contains three high level functions (<samp>match_strings</samp>, <samp>match_most_similar</samp> and <samp>group_similar_strings</samp>) that can be used directly, and one class (<samp>StringGrouper</samp>) that allows for a more iterative approach. 

The permitted calling patterns of the three functions, and their return types, are:

| Function        | Parameters | <samp>pandas</samp> Return Type |
| -------------: |:-------------|:-----:|
| <samp>match_strings</samp>| <samp>(master, **kwargs)</samp>| <samp>DataFrame</samp> |
| <samp>match_strings</samp>| <samp>(master, duplicates, **kwargs)</samp>| <samp>DataFrame</samp> |
| <samp>match_strings</samp>| <samp>(master, master_id=id_series, **kwargs)</samp>| <samp>DataFrame</samp> |
| <samp>match_strings</samp>| <samp>(master, duplicates, master_id, duplicates_id, **kwargs)</samp>| <samp>DataFrame</samp> |
| <samp>match_most_similar</samp>| <samp>(master, duplicates, **kwargs)</samp>| <samp>Series</samp> (if kwarg `ignore_index=True`) otherwise <samp>DataFrame</samp> (default)|
| <samp>match_most_similar</samp>| <samp>(master, duplicates, master_id, duplicates_id, **kwargs)</samp>| <samp>DataFrame</samp> |
| <samp>group_similar_strings</samp>| <samp>(strings_to_group, **kwargs)</samp>| <samp>Series</samp> (if kwarg `ignore_index=True`) otherwise <samp>DataFrame</samp> (default)|
| <samp>group_similar_strings</samp>| <samp>(strings_to_group, strings_id, **kwargs)</samp>| <samp>DataFrame</samp> |
| <samp>compute_pairwise_similarities</samp>| <samp>(string_series_1, string_series_2, **kwargs)</samp>| <samp>Series</samp> |

In the rest of this document the names, <samp>Series</samp> and <samp>DataFrame</samp>, refer to the familiar <samp>pandas</samp> object types.
#### Parameters:

|Name | Description |
|:--- | :--- |
|**<samp>master</samp>** | A <samp>Series</samp> of strings to be matched with themselves (or with those in <samp>duplicates</samp>). |
|**<samp>duplicates</samp>** | A <samp>Series</samp> of strings to be matched with those of <samp>master</samp>. |
|**<samp>master_id</samp>** (or <samp>id_series</samp>) | A <samp>Series</samp> of IDs corresponding to the strings in <samp>master</samp>. |
|**<samp>duplicates_id</samp>** | A <samp>Series</samp> of IDs corresponding to the strings in <samp>duplicates</samp>. |
|**<samp>strings_to_group</samp>** | A <samp>Series</samp> of strings to be grouped. |
|**<samp>strings_id</samp>** | A <samp>Series</samp> of IDs corresponding to the strings in <samp>strings_to_group</samp>. |
|**<samp>string_series_1(_2)</samp>** | A <samp>Series</samp> of strings each of which is to be compared with its corresponding string in <samp>string_series_2(_1)</samp>. |
|**<samp>**kwargs</samp>** | Keyword arguments (see [below](#kwargs)).|

#### Functions:

* #### `match_strings` 
   Returns a <samp>DataFrame</samp> containing similarity-scores of all matching pairs of highly similar strings (from the input <samp>Series</samp> <samp>master</samp> and <samp>duplicates</samp>).  The column-names of the output are a concatenation of three sets:
   1. The name of the Series <samp>master</samp> and the name(s) of its index(es) prefixed by the string `'left_'`,
   2. `'similarity'` containing the similarity-scores, and 
   3. The name of the Series <samp>duplicates</samp> (or <samp>master</samp> if <samp>duplicates</samp> is not given) and the name(s) of its index(es) prefixed by the string `'right_'`.
   
   If any of the input <samp>Series</samp> has no name, it assumes the name `'side'` and is prefixed as described above.  Similarly, if any of the indexes has no name it assumes its <samp>pandas</samp> default name (`'index'`, `'level_0'`, and so on) and is prefixed as described above.
   
   If only parameter <samp>master</samp> is given, it will return pairs of highly similar strings within <samp>master</samp>.    This can be seen as a self-join (both <samp>'left_'</samp> and <samp>'right_'</samp> prefixed columns come from <samp>master</samp>). If both parameters <samp>master</samp> and <samp>duplicates</samp> are given, it will return pairs of highly similar strings between <samp>master</samp> and <samp>duplicates</samp>. This can be seen as an inner-join (<samp>'left_'</samp> and <samp>'right_'</samp> prefixed columns come from <samp>master</samp> and <samp>duplicates</samp> respectively).     
   
   The function also supports optionally inputting IDs (<samp>master_id</samp> and <samp>duplicates_id</samp>) corresponding to the string values being matched. In which case, the output includes two additional columns whose names are the names of these optional Series prefixed by <samp>'left_'</samp> and <samp>'right_'</samp> accordingly, and containing the IDs corresponding to the strings in the output.  If any of these <samp>Series</samp> has no name, then it assumes the name `'id'` and is prefixed as described above.
   
   If no index-columns are desired in the output, the keyword argument setting `ignore_index=True` will exclude all the index-columns.  (See [tutorials/ignore_index_and_replace_na.md](tutorials/ignore_index_and_replace_na.md) for a demonstration.)
   
   
* #### `match_most_similar` 
   If `ignore_index=True`, returns a <samp>Series</samp> of strings, where for each string in <samp>duplicates</samp> the most similar string in <samp>master</samp> is returned.  If there are no similar strings in <samp>master</samp> for a given string in <samp>duplicates</samp> (because there is no potential match where the cosine similarity is above the threshold (default: 0.8)) then the original string in <samp>duplicates</samp> is returned.  The output <samp>Series</samp> thus has the same length and index as <samp>duplicates</samp>.  
   
   For example, if an input <samp>Series</samp> with the contents <samp>\[foooo, bar, baz\]</samp> is passed as the argument to <samp>master</samp>, and <samp>\[foooob, bar, new\]</samp> as the argument to <samp>duplicates</samp>, the function will return: <samp>[foooo, bar, new]</samp>.
   
   The name of the output <samp>Series</samp> is the same as that of <samp>master</samp> prefixed with the string `'most_similar_'`.  If <samp>master</samp> has no name, it is assumed to have the name `'master'` before being prefixed.
       
   If `ignore_index=False` (the default), `match_most_similar` returns a <samp>DataFrame</samp> containing the same <samp>Series</samp> described above as one of its columns.  So it also inherits the same index and length as <samp>duplicates</samp>.  The rest of its columns correspond to the index(es) of <samp>master</samp> and thus contain the index-labels of the most similar strings being output.  If there are no similar strings in <samp>master</samp> for a given string in <samp>duplicates</samp> then the value assigned to these index-columns is `NaN` by default.  However, if the keyword argument `replace_na=True`, then these `NaN` values are replaced with the index-label(s) of the corresponding string in <samp>duplicates</samp>.  Note that such replacements can only occur if the indexes of <samp>master</samp> and <samp>duplicates</samp> have the same number of levels.  (See [tutorials/ignore_index_and_replace_na.md](tutorials/ignore_index_and_replace_na.md#MMS) for a demonstration.)
   
   Each column-name of the output <samp>DataFrame</samp> has the same name as its corresponding column, index, or index-level of <samp>master</samp> prefixed with the string `'most_similar_'`.
  
    If both parameters <samp>master_id</samp> and <samp>duplicates_id</samp> are also given, then a <samp>DataFrame</samp> is always returned with the same columns as described above, but with an additional column containing those IDs from these input <samp>Series</samp> corresponding to the output strings.  


* #### `group_similar_strings` 
  Takes a single <samp>Series</samp> of strings (<samp>strings_to_group</samp>) and groups them by assigning to each string one string from <samp>strings_to_group</samp> chosen as the group-representative for each group of similar strings found. (See [tutorials/group_representatives.md](tutorials/group_representatives.md) for details on how the the group-representatives are chosen.)   
  
  If `ignore_index=True`, the output is a <samp>Series</samp> (named `group_rep`) of the same length and index as <samp>strings_to_group</samp> containing the group-representative strings.  
   
  For example, an input Series with contents: <samp>\[foooo, foooob, bar\]</samp> will return <samp>\[foooo, foooo, bar\]</samp>.  Here <samp>foooo</samp> and <samp>foooob</samp> are grouped together into group <samp>foooo</samp> because they are found to be similar.  Another example can be found [below](#dedup).
  
   If `ignore_index=False`, the output is a <samp>DataFrame</samp> containing the above <samp>Series</samp> (named `group_rep`) as one of its columns.  The remaining column(s) correspond to the index(es) of <samp>strings_to_group</samp> and contain the index-labels of the group-representatives.
   
   If <samp>strings_id</samp> is also given, then the IDs from <samp>strings_id</samp> corresponding to the group-representatives is also returned.  
   

* #### `compute_pairwise_similarities`
   Returns a <samp>Series</samp> of cosine similarity scores the same length as <samp>string_series_1</samp> and <samp>string_series_2</samp>.  Each score is the cosine similarity between its corresponding strings in the two input <samp>Series</samp>.
   

All functions are built using a class **<samp>StringGrouper</samp>**. This class can be used through pre-defined functions, for example the three high level functions above, as well as using a more iterative approach where matches can be added or removed if needed by calling the **<samp>StringGrouper</samp>** class directly.
   

#### Options:

* #### <a name="kwargs"></a>`kwargs`

   All keyword arguments not mentioned in the function definitions above are used to update the default settings. The following optional arguments can be used:

   * **<samp>ngram_size</samp>**: The amount of characters in each n-gram. Optional. Default is <samp>3</samp>
   * **<samp>regex</samp>**: The regex string used to clean-up the input string. Optional. Default is <samp>"[,-./]|\s"</samp>.
   * **<samp>max_n_matches</samp>**: The maximum number of matches allowed per string. Default is <samp>20</samp>.
   * **<samp>min_similarity</samp>**: The minimum cosine similarity for two strings to be considered a match.
    Defaults to <samp>0.8</samp>
   * **<samp>number_of_processes</samp>**: The number of processes used by the cosine similarity calculation. Defaults to
    `number of cores on a machine - 1.`
   * **<samp>ignore_index</samp>**: Determines whether indexes are ignored or not.  If `False` (the default), index-columns will appear in the output, otherwise not.
   * **<samp>replace_na</samp>**: For function <samp>match_most_similar</samp>, determines whether `NaN` values in index-columns are replaced or not by index-labels from <samp>duplicates</samp>. Defaults to `False`.  
   * **<samp>group_rep</samp>**: For function <samp>group_similar_strings</samp>, determines how group-representatives are chosen.  Allowed values are `'centroid'` (the default) and `'first'`.  See [tutorials/group_representatives.md](tutorials/group_representatives.md) for an explanation.

## Examples

In this section we will cover a few use cases for which string_grouper may be used. We will use the same data set of company names as used in: [Super Fast String Matching in Python](https://bergvca.github.io/2017/10/14/super-fast-string-matching.html).

### Find all matches within a single data set


```python
import pandas as pd
import numpy as np
from string_grouper import match_strings, match_most_similar, \
	group_similar_strings, compute_pairwise_similarities, \
	StringGrouper
```


```python
company_names = '/media/chris/data/dev/name_matching/data/sec_edgar_company_info.csv'
# We only look at the first 50k as an example:
companies = pd.read_csv(company_names)[0:50000]
# Create all matches:
matches = match_strings(companies['Company Name'])
# Look at only the non-exact matches:
matches[matches['left_Company Name'] != matches['right_Company Name']].head()
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_index</th>
      <th>left_Company Name</th>
      <th>similarity</th>
      <th>right_Company Name</th>
      <th>right_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>14</td>
      <td>0210, LLC</td>
      <td>0.870291</td>
      <td>90210 LLC</td>
      <td>4211</td>
    </tr>
    <tr>
      <th>167</th>
      <td>165</td>
      <td>1 800 MUTUALS ADVISOR SERIES</td>
      <td>0.931615</td>
      <td>1 800 MUTUALS ADVISORS SERIES</td>
      <td>166</td>
    </tr>
    <tr>
      <th>168</th>
      <td>166</td>
      <td>1 800 MUTUALS ADVISORS SERIES</td>
      <td>0.931615</td>
      <td>1 800 MUTUALS ADVISOR SERIES</td>
      <td>165</td>
    </tr>
    <tr>
      <th>172</th>
      <td>168</td>
      <td>1 800 RADIATOR FRANCHISE INC</td>
      <td>1.000000</td>
      <td>1-800-RADIATOR FRANCHISE INC.</td>
      <td>201</td>
    </tr>
    <tr>
      <th>178</th>
      <td>173</td>
      <td>1 FINANCIAL MARKETPLACE SECURITIES LLC        ...</td>
      <td>0.949364</td>
      <td>1 FINANCIAL MARKETPLACE SECURITIES, LLC</td>
      <td>174</td>
    </tr>
  </tbody>
</table>
</div>


### Find all matches in between two data sets. 
The <samp>match_strings</samp> function finds similar items between two data sets as well. This can be seen as an inner join between two data sets:


```python
# Create a small set of artificial company names:
duplicates = pd.Series(['S MEDIA GROUP', '012 SMILE.COMMUNICATIONS', 'foo bar', 'B4UTRADE COM CORP'])
# Create all matches:
matches = match_strings(companies['Company Name'], duplicates)
matches
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_index</th>
      <th>left_Company Name</th>
      <th>similarity</th>
      <th>right_side</th>
      <th>right_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>012 SMILE.COMMUNICATIONS LTD</td>
      <td>0.944092</td>
      <td>012 SMILE.COMMUNICATIONS</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49777</td>
      <td>B.A.S. MEDIA GROUP</td>
      <td>0.854383</td>
      <td>S MEDIA GROUP</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49855</td>
      <td>B4UTRADE COM CORP</td>
      <td>1.000000</td>
      <td>B4UTRADE COM CORP</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49856</td>
      <td>B4UTRADE COM INC</td>
      <td>0.810217</td>
      <td>B4UTRADE COM CORP</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49857</td>
      <td>B4UTRADE CORP</td>
      <td>0.878276</td>
      <td>B4UTRADE COM CORP</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


Out of the four company names in <samp>duplicates</samp>, three companies are found in the original company data set. One company is found three times.

### Finding duplicates from a (database extract to) DataFrame where IDs for rows are supplied.

A very common scenario is the case where duplicate records for an entity have been entered into a database. That is, there are two or more records where a name field has slightly different spelling. For example, "A.B. Corporation" and "AB Corporation". Using the optional 'ID' parameter in the <samp>match_strings</samp> function duplicates can be found easily. A [tutorial](tutorials/tutorial_1.md) that steps though the process with an example data set is available.


### For a second data set, find only the most similar match

In the example above, it's possible that multiple matches are found for a single string. Sometimes we just want a string to match with a single most similar string. If there are no similar strings found, the original string should be returned:


```python
# Create a small set of artificial company names:
new_companies = pd.Series(['S MEDIA GROUP', '012 SMILE.COMMUNICATIONS', 'foo bar', 'B4UTRADE COM CORP'])
# Create all matches:
matches = match_most_similar(companies['Company Name'], new_companies)
# Display the results:
pd.DataFrame({'new_companies': new_companies, 'duplicates': matches})
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>New Company</th>
      <th>most_similar_Company Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S MEDIA GROUP</td>
      <td>B.A.S. MEDIA GROUP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>012 SMILE.COMMUNICATIONS</td>
      <td>012 SMILE.COMMUNICATIONS LTD</td>
    </tr>
    <tr>
      <th>2</th>
      <td>foo bar</td>
      <td>foo bar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B4UTRADE COM CORP</td>
      <td>B4UTRADE COM CORP</td>
    </tr>
  </tbody>
</table>
</div>



### <a name="dedup"></a>Deduplicate a single data set and show items with most duplicates

The <samp>group_similar_strings</samp> function groups strings that are similar using a single linkage clustering algorithm. That is, if item A and item B are similar; and item B and item C are similar; but the similarity between A and C is below the threshold; then all three items are grouped together. 

```python
# Add the grouped strings:
companies['deduplicated_name'] = group_similar_strings(companies['Company Name'],
                                                       ignore_index=True)
# Show items with most duplicates:
companies.groupby('deduplicated_name')['Line Number'].count().sort_values(ascending=False).head(10)
```




    deduplicated_name
    ADVISORS DISCIPLINED TRUST                                      1824
    AGL LIFE ASSURANCE CO SEPARATE ACCOUNT                           183
    ANGELLIST-ART-FUND, A SERIES OF ANGELLIST-FG-FUNDS, LLC          116
    AMERICREDIT AUTOMOBILE RECEIVABLES TRUST 2001-1                   87
    ACE SECURITIES CORP. HOME EQUITY LOAN TRUST, SERIES 2006-HE2      57
    ASSET-BACKED PASS-THROUGH CERTIFICATES SERIES 2004-W1             40
    ALLSTATE LIFE GLOBAL FUNDING TRUST 2005-3                         39
    ALLY AUTO RECEIVABLES TRUST 2014-1                                33
    ANDERSON ROBERT E /                                               28
    ADVENT INTERNATIONAL GPE VIII LIMITED PARTNERSHIP                 28
    Name: Line Number, dtype: int64


The <samp>group_similar_strings</samp> function also works with IDs: imagine a <samp>DataFrame</samp> (<samp>customers_df</samp>) with the following content:
```python
# Create a small set of artificial customer names:
customers_df = pd.DataFrame(
   [
      ('BB016741P', 'Mega Enterprises Corporation'),
      ('CC082744L', 'Hyper Startup Incorporated'),
      ('AA098762D', 'Hyper Startup Inc.'),
      ('BB099931J', 'Hyper-Startup Inc.'),
      ('HH072982K', 'Hyper Hyper Inc.')
   ],
   columns=('Customer ID', 'Customer Name')
).set_index('Customer ID')
# Display the data:
customers_df
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer Name</th>
    </tr>
    <tr>
      <th>Customer ID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BB016741P</th>
      <td>Mega Enterprises Corporation</td>
    </tr>
    <tr>
      <th>CC082744L</th>
      <td>Hyper Startup Incorporated</td>
    </tr>
    <tr>
      <th>AA098762D</th>
      <td>Hyper Startup Inc.</td>
    </tr>
    <tr>
      <th>BB099931J</th>
      <td>Hyper-Startup Inc.</td>
    </tr>
    <tr>
      <th>HH072982K</th>
      <td>Hyper Hyper Inc.</td>
    </tr>
  </tbody>
</table>
</div>

The output of <samp>group_similar_strings</samp> can be directly used as a mapping table:
```python
# Group customers with similar names:
customers_df[["group-id", "name_deduped"]]  = \
    group_similar_strings(customers_df["Customer Name"])
# Display the mapping table:
customers_df
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer Name</th>
      <th>group-id</th>
      <th>name_deduped</th>
    </tr>
    <tr>
      <th>Customer ID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BB016741P</th>
      <td>Mega Enterprises Corporation</td>
      <td>BB016741P</td>
      <td>Mega Enterprises Corporation</td>
    </tr>
    <tr>
      <th>CC082744L</th>
      <td>Hyper Startup Incorporated</td>
      <td>CC082744L</td>
      <td>Hyper Startup Incorporated</td>
    </tr>
    <tr>
      <th>AA098762D</th>
      <td>Hyper Startup Inc.</td>
      <td>AA098762D</td>
      <td>Hyper Startup Inc.</td>
    </tr>
    <tr>
      <th>BB099931J</th>
      <td>Hyper-Startup Inc.</td>
      <td>AA098762D</td>
      <td>Hyper Startup Inc.</td>
    </tr>
    <tr>
      <th>HH072982K</th>
      <td>Hyper Hyper Inc.</td>
      <td>HH072982K</td>
      <td>Hyper Hyper Inc.</td>
    </tr>
  </tbody>
</table>
</div>

Note that here <samp>customers_df</samp> initially had only two columns "Customer ID" and "Customer Name" (before the <samp>group_similar_strings</samp> function call); and it acquired two more columns "group-id" and "name_deduped" after the call.


## The StringGrouper class

The three functions mentioned above all create a <samp>StringGrouper</samp> object behind the scenes and call different functions on it. The <samp>StringGrouper</samp> class keeps track of all tuples of similar strings and creates the groups out of these. Since matches are often not perfect, a common workflow is to:

1. Create matches
2. Manually inspect the results
3. Add and remove matches where necessary
4. Create groups of similar strings

The <samp>StringGrouper</samp> class allows for this without having to re-calculate the cosine similarity matrix. See below for an example. 


```python
company_names = '/media/chris/data/dev/name_matching/data/sec_edgar_company_info.csv'
companies = pd.read_csv(company_names)[:50000]
```

1. Create matches


```python
# Create a new StringGrouper
string_grouper = StringGrouper(companies['Company Name'], ignore_index=True)
# Check if the ngram function does what we expect:
string_grouper.n_grams('McDonalds')
```

    ['McD', 'cDo', 'Don', 'ona', 'nal', 'ald', 'lds']


```python
# Now fit the StringGrouper - this will take a while since we are calculating cosine similarities on 600k strings
string_grouper = string_grouper.fit()
```

```python
# Add the grouped strings
companies['deduplicated_name'] = string_grouper.get_groups()
```

Suppose we know that PWC HOLDING CORP and PRICEWATERHOUSECOOPERS LLP are the same company. StringGrouper will not match these since they are not similar enough. 


```python
companies[companies.deduplicated_name.str.contains('PRICEWATERHOUSECOOPERS LLP')]
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Line Number</th>
      <th>Company Name</th>
      <th>Company CIK Key</th>
      <th>deduplicated_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>478441</th>
      <td>478442</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
      <td>1064284</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
    <tr>
      <th>478442</th>
      <td>478443</td>
      <td>PRICEWATERHOUSECOOPERS LLP</td>
      <td>1186612</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
    <tr>
      <th>478443</th>
      <td>478444</td>
      <td>PRICEWATERHOUSECOOPERS SECURITIES LLC</td>
      <td>1018444</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
  </tbody>
</table>
</div>


```python
companies[companies.deduplicated_name.str.contains('PWC')]
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Line Number</th>
      <th>Company Name</th>
      <th>Company CIK Key</th>
      <th>deduplicated_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>485535</th>
      <td>485536</td>
      <td>PWC CAPITAL INC.</td>
      <td>1690640</td>
      <td>PWC CAPITAL INC.</td>
    </tr>
    <tr>
      <th>485536</th>
      <td>485537</td>
      <td>PWC HOLDING CORP</td>
      <td>1456450</td>
      <td>PWC HOLDING CORP</td>
    </tr>
    <tr>
      <th>485537</th>
      <td>485538</td>
      <td>PWC INVESTORS, LLC</td>
      <td>1480311</td>
      <td>PWC INVESTORS, LLC</td>
    </tr>
    <tr>
      <th>485538</th>
      <td>485539</td>
      <td>PWC REAL ESTATE VALUE FUND I LLC</td>
      <td>1668928</td>
      <td>PWC REAL ESTATE VALUE FUND I LLC</td>
    </tr>
    <tr>
      <th>485539</th>
      <td>485540</td>
      <td>PWC SECURITIES CORP                                     /BD</td>
      <td>1023989</td>
      <td>PWC SECURITIES CORP                                     /BD</td>
    </tr>
    <tr>
      <th>485540</th>
      <td>485541</td>
      <td>PWC SECURITIES CORPORATION</td>
      <td>1023989</td>
      <td>PWC SECURITIES CORPORATION</td>
    </tr>
    <tr>
      <th>485541</th>
      <td>485542</td>
      <td>PWCC LTD</td>
      <td>1172241</td>
      <td>PWCC LTD</td>
    </tr>
    <tr>
      <th>485542</th>
      <td>485543</td>
      <td>PWCG BROKERAGE, INC.</td>
      <td>67301</td>
      <td>PWCG BROKERAGE, INC.</td>
    </tr>
  </tbody>
</table>
</div>


We can add these with the add function:


```python
string_grouper = string_grouper.add_match('PRICEWATERHOUSECOOPERS LLP', 'PWC HOLDING CORP')
companies['deduplicated_name'] = string_grouper.get_groups()
# Now lets check again:

companies[companies.deduplicated_name.str.contains('PRICEWATERHOUSECOOPERS LLP')]
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Line Number</th>
      <th>Company Name</th>
      <th>Company CIK Key</th>
      <th>deduplicated_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>478441</th>
      <td>478442</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
      <td>1064284</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
    <tr>
      <th>478442</th>
      <td>478443</td>
      <td>PRICEWATERHOUSECOOPERS LLP</td>
      <td>1186612</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
    <tr>
      <th>478443</th>
      <td>478444</td>
      <td>PRICEWATERHOUSECOOPERS SECURITIES LLC</td>
      <td>1018444</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
    <tr>
      <th>485536</th>
      <td>485537</td>
      <td>PWC HOLDING CORP</td>
      <td>1456450</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
  </tbody>
</table>
</div>


This can also be used to merge two groups:


```python
string_grouper = string_grouper.add_match('PRICEWATERHOUSECOOPERS LLP', 'ZUCKER MICHAEL')
companies['deduplicated_name'] = string_grouper.get_groups()

# Now lets check again:
companies[companies.deduplicated_name.str.contains('PRICEWATERHOUSECOOPERS LLP')]
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Line Number</th>
      <th>Company Name</th>
      <th>Company CIK Key</th>
      <th>deduplicated_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>478441</th>
      <td>478442</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
      <td>1064284</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
    <tr>
      <th>478442</th>
      <td>478443</td>
      <td>PRICEWATERHOUSECOOPERS LLP</td>
      <td>1186612</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
    <tr>
      <th>478443</th>
      <td>478444</td>
      <td>PRICEWATERHOUSECOOPERS SECURITIES LLC</td>
      <td>1018444</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
    <tr>
      <th>485536</th>
      <td>485537</td>
      <td>PWC HOLDING CORP</td>
      <td>1456450</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
    <tr>
      <th>662585</th>
      <td>662586</td>
      <td>ZUCKER MICHAEL</td>
      <td>1629018</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
    <tr>
      <th>662604</th>
      <td>662605</td>
      <td>ZUCKERMAN MICHAEL</td>
      <td>1303321</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
    <tr>
      <th>662605</th>
      <td>662606</td>
      <td>ZUCKERMAN MICHAEL</td>
      <td>1496366</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
  </tbody>
</table>
</div>


We can remove strings from groups in the same way:


```python
string_grouper = string_grouper.remove_match('PRICEWATERHOUSECOOPERS LLP', 'ZUCKER MICHAEL')
companies['deduplicated_name'] = string_grouper.get_groups()

# Now lets check again:
companies[companies.deduplicated_name.str.contains('PRICEWATERHOUSECOOPERS LLP')]
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Line Number</th>
      <th>Company Name</th>
      <th>Company CIK Key</th>
      <th>deduplicated_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>478441</th>
      <td>478442</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
      <td>1064284</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
    <tr>
      <th>478442</th>
      <td>478443</td>
      <td>PRICEWATERHOUSECOOPERS LLP</td>
      <td>1186612</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
    <tr>
      <th>478443</th>
      <td>478444</td>
      <td>PRICEWATERHOUSECOOPERS SECURITIES LLC</td>
      <td>1018444</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
    <tr>
      <th>485536</th>
      <td>485537</td>
      <td>PWC HOLDING CORP</td>
      <td>1456450</td>
      <td>PRICEWATERHOUSECOOPERS LLP                              /TA</td>
    </tr>
  </tbody>
</table>
</div>
