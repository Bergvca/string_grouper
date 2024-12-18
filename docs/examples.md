
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
The `match_strings` function finds similar items between two data sets as well. This can be seen as an inner join between two data sets:


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


Out of the four company names in `duplicates`, three companies are found in the original company data set. One company is found three times.

### Finding duplicates from a (database extract to) DataFrame where IDs for rows are supplied.

A very common scenario is the case where duplicate records for an entity have been entered into a database. That is, there are two or more records where a name field has slightly different spelling. For example, "A.B. Corporation" and "AB Corporation". Using the optional 'ID' parameter in the `match_strings` function duplicates can be found easily. A [tutorial](https://github.com/Bergvca/string_grouper/blob/master/tutorials/tutorial_1.md) that steps though the process with an example data set is available.


### For a second data set, find only the most similar match

In the example above, it's possible that multiple matches are found for a single string. Sometimes we just want a string to match with a single most similar string. If there are no similar strings found, the original string should be returned:


```python
# Create a small set of artificial company names:
new_companies = pd.Series(['S MEDIA GROUP', '012 SMILE.COMMUNICATIONS', 'foo bar', 'B4UTRADE COM CORP'],\
                          name='New Company')
# Create all matches:
matches = match_most_similar(companies['Company Name'], new_companies, ignore_index=True)
# Display the results:
pd.concat([new_companies, matches], axis=1)
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

The `group_similar_strings` function groups strings that are similar using a single linkage clustering algorithm. That is, if item A and item B are similar; and item B and item C are similar; but the similarity between A and C is below the threshold; then all three items are grouped together. 

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


The `group_similar_strings` function also works with IDs: imagine a `DataFrame` (`customers_df`) with the following content:
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

The output of `group_similar_strings` can be directly used as a mapping table:
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

Note that here `customers_df` initially had only one column "Customer Name" (before the `group_similar_strings` function call); and it acquired two more columns "group-id" (the index-column) and "name_deduped" after the call through a "[setting with enlargement](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#setting-with-enlargement)" (a `pandas` feature).

### <a name="dot"></a>Simply compute the cosine similarities of pairs of strings

Sometimes we have pairs of strings that have already been matched but whose similarity scores need to be computed.  For this purpose we provide the function `compute_pairwise_similarities`:

```python
# Create a small DataFrame of pairs of strings:
pair_s = pd.DataFrame(
    [
        ('Mega Enterprises Corporation', 'Mega Enterprises Corporation'),
        ('Hyper Startup Inc.', 'Hyper Startup Incorporated'),
        ('Hyper Startup Inc.', 'Hyper Startup Inc.'),
        ('Hyper Startup Inc.', 'Hyper-Startup Inc.'),
        ('Hyper Hyper Inc.', 'Hyper Hyper Inc.'),
        ('Mega Enterprises Corporation', 'Mega Enterprises Corp.')
   ],
   columns=('left', 'right')
)
# Display the data:
pair_s
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left</th>
      <th>right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mega Enterprises Corporation</td>
      <td>Mega Enterprises Corporation</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hyper Startup Inc.</td>
      <td>Hyper Startup Incorporated</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hyper Startup Inc.</td>
      <td>Hyper Startup Inc.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hyper Startup Inc.</td>
      <td>Hyper-Startup Inc.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hyper Hyper Inc.</td>
      <td>Hyper Hyper Inc.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Mega Enterprises Corporation</td>
      <td>Mega Enterprises Corp.</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Compute their cosine similarities and display them:
pair_s['similarity'] = compute_pairwise_similarities(pair_s['left'], pair_s['right'])
pair_s
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left</th>
      <th>right</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mega Enterprises Corporation</td>
      <td>Mega Enterprises Corporation</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hyper Startup Inc.</td>
      <td>Hyper Startup Incorporated</td>
      <td>0.633620</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hyper Startup Inc.</td>
      <td>Hyper Startup Inc.</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hyper Startup Inc.</td>
      <td>Hyper-Startup Inc.</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hyper Hyper Inc.</td>
      <td>Hyper Hyper Inc.</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Mega Enterprises Corporation</td>
      <td>Mega Enterprises Corp.</td>
      <td>0.826463</td>
    </tr>
  </tbody>
</table>
</div>

