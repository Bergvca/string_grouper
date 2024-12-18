---
title: String Grouper Class
---

## Concept 

All functions are built using a class **`StringGrouper`**. This class can be used through pre-defined functions, for example the four high level functions above, as well as using a more interactive approach where matches can be added or removed if needed by calling the **`StringGrouper`** class directly.
   

The four functions mentioned above all create a `StringGrouper` object behind the scenes and call different functions on it. The `StringGrouper` class keeps track of all tuples of similar strings and creates the groups out of these. Since matches are often not perfect, a common workflow is to:

## Example

1. Create matches
2. Manually inspect the results
3. Add and remove matches where necessary
4. Create groups of similar strings

The `StringGrouper` class allows for this without having to re-calculate the cosine similarity matrix. See below for an example. 


```python
company_names = './data/sec_edgar_company_info.csv'
companies = pd.read_csv(company_names)
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
string_grouper.n_grams('ÀbracâDABRÀ')
```
  
    ['abr', 'bra', 'rac', 'aca', 'cad', 'ada', 'dab', 'abr', 'bra']

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

