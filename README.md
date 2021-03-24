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
from string_grouper import match_strings, match_most_similar, group_similar_strings, StringGrouper
```

As shown above, the library may be used together with <samp>pandas</samp>, and contains three high level functions (<samp>match_strings</samp>, <samp>match_most_similar</samp> and <samp>group_similar_strings</samp>) that can be used directly, and one class (<samp>StringGrouper</samp>) that allows for a more iterative approach. 

The permitted calling patterns of the three functions, and their return types, are:

| Function        | Parameters | <samp>pandas</samp> Return Type |
| -------------: |:-------------|:-----:|
| <samp>match_strings</samp>| <samp>(master, **kwargs)</samp>| <samp>DataFrame</samp> |
| <samp>match_strings</samp>| <samp>(master, duplicates, **kwargs)</samp>| <samp>DataFrame</samp> |
| <samp>match_strings</samp>| <samp>(master, master_id=id_series, **kwargs)</samp>| <samp>DataFrame</samp> |
| <samp>match_strings</samp>| <samp>(master, duplicates, master_id, duplicates_id, **kwargs)</samp>| <samp>DataFrame</samp> |
| <samp>match_most_similar</samp>| <samp>(master, duplicates, **kwargs)</samp>| <samp>Series</samp> |
| <samp>match_most_similar</samp>| <samp>(master, duplicates, master_id, duplicates_id, **kwargs)</samp>| <samp>DataFrame</samp> |
| <samp>group_similar_strings</samp>| <samp>(strings_to_group, **kwargs)</samp>| <samp>Series</samp> |
| <samp>group_similar_strings</samp>| <samp>(strings_to_group, strings_id, **kwargs)</samp>| <samp>DataFrame</samp> |

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
|**<samp>**kwargs</samp>** | Keyword arguments (see [below](#kwargs)).|

#### Functions:

* #### `match_strings` 
   Returns all pairs of highly similar strings in a <samp>DataFrame</samp>.  The column names of the output  <samp>DataFrame</samp> are 'left_side', 'right_side' and 'similarity'. 
   
   If only parameter <samp>master</samp> is given, it will return pairs of highly similar strings within <samp>master</samp>.    This can be seen as a self-join (both 'left_side' and 'right_side' column values come from <samp>master</samp>). If both parameters <samp>master</samp> and <samp>duplicates</samp> are given, it will return pairs of highly similar strings between <samp>master</samp> and <samp>duplicates</samp>. This can be seen as an inner-join ('left_side' and 'right_side' column values come from <samp>master</samp> and <samp>duplicates</samp> respectively).     
   
   The function also supports optionally inputting IDs (<samp>master_id</samp> and <samp>duplicates_id</samp>) corresponding to the string values being matched. In which case, the output includes two additional columns whose names are 'left_side_id' and 'right_side_id' containing the IDs corresponding to the string values in 'left_side' and 'right_side' respectively.  
   
   
* #### `match_most_similar` 
   Returns a nameless <samp>Series</samp> of strings of the same length as the parameter <samp>duplicates</samp>, where for each string in <samp>duplicates</samp> the most similar string in <samp>master</samp> is returned. If there are no similar strings in <samp>master</samp> for a given string in <samp>duplicates</samp>
    (there is no potential match where the cosine similarity is above the threshold (default: 0.8)) 
    the original string in <samp>duplicates</samp> is returned.
  
   For example, if the input series <samp>\[foooo, bar, baz\]</samp> is passed as the argument to <samp>master</samp>, and <samp>\[foooob, bar, new\]</samp> as the argument to <samp>duplicates</samp>, the function will return:
    <samp>[foooo, bar, new]</samp>.
    
    If both parameters <samp>master_id</samp> and <samp>duplicates_id</samp> are also given, then a <samp>DataFrame</samp> with two unnamed columns is returned.  The second column is the same as the <samp>Series</samp> of strings described above, and the first column contains the corresponding IDs. 
    
* #### `group_similar_strings` 
  Takes a single <samp>Series</samp> (<samp>strings_to_group</samp>) of strings and groups them by assigning to each string one single string chosen as the group-representative (see [string_grouper_utils](tutorials/group_representatives.md)) for each group of similar strings found.   The output is a nameless <samp>Series</samp> of group-representative strings of the same length as the input <samp>Series</samp>.  
   
   For example, the input series: <samp>[foooo, foooob, bar]</samp> will return <samp>[foooo, foooo, bar]</samp>. Here <samp>foooo</samp> and <samp>foooob</samp> are grouped together into group <samp>foooo</samp> because they are found to be similar. (Another example can be found [here](#dedup).)
   
   If <samp>strings_id</samp> is also given, then the IDs corresponding to the output <samp>Series</samp> above is also returned.  The combined output is a <samp>DataFrame</samp> with two columns.
   
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

## Examples

In this section we will cover a few use cases for which string_grouper may be used. We will use the same data set of company names as used in: [Super Fast String Matching in Python](https://bergvca.github.io/2017/10/14/super-fast-string-matching.html).

### Find all matches within a single data set


```python
import pandas as pd
import numpy as np
from string_grouper import match_strings, match_most_similar, group_similar_strings, StringGrouper
```


```python
company_names = '/media/chris/data/dev/name_matching/data/sec_edgar_company_info.csv'
# We only look at the first 50k as an example:
companies = pd.read_csv(company_names)[0:50000]
# Create all matches:
matches = match_strings(companies['Company Name'])
# Look at only the non-exact matches:
matches[matches.left_side != matches.right_side].head()
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_side</th>
      <th>right_side</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>0210, LLC</td>
      <td>90210 LLC</td>
      <td>0.870291</td>
    </tr>
    <tr>
      <th>167</th>
      <td>1 800 MUTUALS ADVISOR SERIES</td>
      <td>1 800 MUTUALS ADVISORS SERIES</td>
      <td>0.931616</td>
    </tr>
    <tr>
      <th>169</th>
      <td>1 800 MUTUALS ADVISORS SERIES</td>
      <td>1 800 MUTUALS ADVISOR SERIES</td>
      <td>0.931616</td>
    </tr>
    <tr>
      <th>171</th>
      <td>1 800 RADIATOR FRANCHISE INC</td>
      <td>1-800-RADIATOR FRANCHISE INC.</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>178</th>
      <td>1 FINANCIAL MARKETPLACE SECURITIES LLC        ...</td>
      <td>1 FINANCIAL MARKETPLACE SECURITIES, LLC</td>
      <td>0.949364</td>
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
      <th>left_side</th>
      <th>right_side</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>012 SMILE.COMMUNICATIONS LTD</td>
      <td>012 SMILE.COMMUNICATIONS</td>
      <td>0.944092</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B.A.S. MEDIA GROUP</td>
      <td>S MEDIA GROUP</td>
      <td>0.854383</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B4UTRADE COM CORP</td>
      <td>B4UTRADE COM CORP</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B4UTRADE COM INC</td>
      <td>B4UTRADE COM CORP</td>
      <td>0.810217</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B4UTRADE CORP</td>
      <td>B4UTRADE COM CORP</td>
      <td>0.878276</td>
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
      <th>new_companies</th>
      <th>duplicates</th>
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
companies['deduplicated_name'] = group_similar_strings(companies['Company Name'])
# Show items with most duplicates:
companies.groupby('deduplicated_name').count().sort_values('Line Number', ascending=False).head(10)['Line Number']

```




    deduplicated_name
    ADVISORS DISCIPLINED TRUST 1100                        188
    ACE SECURITIES CORP HOME EQUITY LOAN TRUST 2005-HE4     32
    AMERCREDIT AUTOMOBILE RECEIVABLES TRUST 2010-1          28
    ADVENT LATIN AMERICAN PRIVATE EQUITY FUND II-A CV       25
    ALLSTATE LIFE GLOBAL FUNDING TRUST 2004-1               24
    ADVENT INTERNATIONAL GPE VII LIMITED PARTNERSHIP        24
    7ADVISORS DISCIPLINED TRUST 1197                        23
    AMERICREDIT AUTOMOBILE RECEIVABLES TRUST  2002 - D      23
    ALLY AUTO RECEIVABLES TRUST 2010-1                      23
    ANDERSON DAVID  A                                       23
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
)
# Display the data:
customers_df
```

|	|Customer ID  | Customer Name|
|---|---|---|
|0  |BB016741P  |Mega Enterprises Corporation|
|1	|CC082744L	|Hyper Startup Incorporated|
|2	|AA098762D	|Hyper Startup Inc.|
|3	|BB099931J	|Hyper-Startup Inc.|
|4	|HH072982K	|Hyper Hyper Inc.|

The output of <samp>group_similar_strings</samp> can be directly used as a mapping table:
```python
# Group customers with similar names:
customers_df[["group-id", "name_deduped"]]  = \
    group_similar_strings(customers_df["Customer Name"], customers_df["Customer ID"])
# Display the mapping table:
customers_df
```

Customer ID | Customer Name | group-id | name_deduped 
-- | -- | -- | --
BB016741P | Mega Enterprises Corporation | BB016741P | Mega Enterprises Corporation 
CC082744L | Hyper Startup Incorporated | CC082744L | Hyper Startup Incorporated 
AA098762D | Hyper Startup Inc. | CC082744L | Hyper Startup Incorporated 
BB099931J | Hyper-Startup Inc. | CC082744L | Hyper Startup Incorporated 
HH072982K | Hyper Hyper Inc. | CC082744L | Hyper Startup Incorporated 

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
# We only look at the first 50k as an example
companies = pd.read_csv(company_names)
```

1. Create matches


```python
# Create a new StringGrouper
string_grouper = StringGrouper(companies['Company Name'])
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
