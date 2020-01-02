
# string_grouper

*string_grouper* is a library that makes finding groups of similar strings within a single or within multiple lists of strings easy. *string_grouper* uses **tf-idf** to calculate **cosine similarities** within a single list or between two lists of strings. The full process is described in the blog [Super Fast String Matching in Python](https://bergvca.github.io/2017/10/14/super-fast-string-matching.html).

The library contains 3 high level functions that can be used directly, and 1 class that allows for a more iterative approach. The three functions are:

* **match_strings**(**master**: pd.Series, **duplicates**: Optional[pd.Series] = None, \**kwargs) -> pd.DataFrame:
Returns all highly similar strings. If only 'master' is given, it will return highly similar strings within master.
    This can be seen as an self-join. If both master and duplicates is given, it will return highly similar strings
    between master and duplicates. This can be seen as an inner-join.
   
   
* **match_most_similar**(**master**: pd.Series, **duplicates**: pd.Series, \**kwargs) -> pd.Series:     Returns a series of strings of the same length as *'duplicates'* where for each string in duplicates the most similar
    string in **'master'** is returned. If there are no similar strings in master for a given string in duplicates
    (there is no potential match where the cosine similarity is above the threshold (default: 0.8)) 
    the original string in duplicates is returned.
  
   For example the input series `[foooo, bar, baz]` (master) and `[foooob, bar, new]` will return:
    `[foooo, bar, new]`
    
    
* **group_similar_strings**(**strings_to_group**: pandas.Series, \**kwargs) -> pandas.Series: Takes a single series of strings and groups these together by picking a single string in each group of similar strings, and return this as output. 
   
   For example the input series: `[foooo, foooob, bar]` will return `[foooo, foooo, bar]`. Here `foooo` and `foooob` are grouped together into group `foooo` because they are found to be similar.
   
All functions are build using a class **StringGrouper**. This class can be used directly as well to allow for a more an more iterative approach where matches can be added or removed if needed. 
   
### kwargs

All keyword arguments not mentioned in the function definition are used to update the default settings. The following optional arguments can be used:

* ***ngram_size***: The amount of characters in each n-gram. Optional. Default is `3`
* ***regex***: The regex string used to cleanup the input string. Optional. Default is `[,-./]|\s`
* ***max_n_matches***: The maximum number of matches allowed per string. Default is `20`.
* ***min_similarity***: The minium cossine similarity for two strings to be considered a match.
    Defaults to `0.8`
* ***number_of_processes***: The number of processes used by the cosine similarity calculation. Defaults to
    `1 - number of cores on a machine.`

## Installing

todo

## Examples

In this section we will cover a few use cases for which string_grouper may be used. We will use the same dataset of company names as used in: [Super Fast String Matching in Python](https://bergvca.github.io/2017/10/14/super-fast-string-matching.html).

### Find all matches within a single dataset


```python
import pandas as pd
import numpy as np
from string_grouper import match_strings, match_most_similar, group_similar_strings, StringGrouper
```


```python
company_names = '/media/chris/data/dev/name_matching/data/sec_edgar_company_info.csv'
# We only look at the first 50k as an example
companies = pd.read_csv(company_names)[0:50000]
# Create all matches:
matches = match_strings(companies['Company Name'])
# Look at only the non-exact matches:
matches[matches.left_side != matches.right_side].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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



### Find all matches in between two datasets. 
The match_string function allows to find similar items between two datasets as well. This can be seen as an inner join between two datasets:



```python
# Create a small set of artifical company names
duplicates = pd.Series(['S MEDIA GROUP', '012 SMILE.COMMUNICATIONS', 'foo bar', 'B4UTRADE COM CORP'])
# Create all matches:
matches = match_strings(companies['Company Name'], duplicates)
matches
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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



Out ouf the 4 company names in `duplicates`, 3 companies are found in the original company dataset. One company is found 3 times.

### For a second dataset, find only the most similar match
In the example above, it's possible that multiple matches are found for a single string. Sometimes we just want a string to match with a single most similar string. If there are no similar strings found, the original string should be returned:



```python
# Create a small set of artifical company names
new_companies = pd.Series(['S MEDIA GROUP', '012 SMILE.COMMUNICATIONS', 'foo bar', 'B4UTRADE COM CORP'])
# Create all matches:
matches = match_most_similar(companies['Company Name'], new_companies)
# Display the results:
pd.DataFrame({'new_companies': new_companies, 'duplicates': matches})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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



### Deduplicate a single dataset and show items with most duplicates

The `group_similar_strings` functions groups strings that are similar using a single linkage clustering algorithm. That is, if item A and item B are similar, and item B and item C are similar but the similarity between A and C is below the threshold, all three items are grouped together. 


```python
# Add the grouped strings
companies['deduplicated_name'] = group_similar_strings(companies['Company Name'])
# Show items with most duplicates
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



## The StringGrouper class

The three functions mentioned above all create a `StringGrouper` object behind the scenes and call different functions on it. The `StringGrouper` class keeps track of all tuples of similar strings and creates the groups out of these. Since matches are often not perfect, a common workflow is to:

1. Create matches
2. Manually inspect the results
3. Add and remove matches were neccesary
4. Create groups of similar Strings

The `StringGrouper` allows for this without having to re-calculate the cosine similarity matrix. See below for an example. 


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


```python
pd.set_option('display.max_colwidth', -1)
```

Suppose we know that PWC HOLDING CORP and PRICEWATERHOUSECOOPERS LLP are the same company. The StringGrouper will not match these, since they are not similar enough. 


```python
companies[companies.deduplicated_name.str.contains('PRICEWATERHOUSECOOPERS LLP')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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


