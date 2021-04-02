```python
import pandas as pd
from string_grouper import match_strings, match_most_similar, group_similar_strings
```

# 1. **match_strings**(..., **ignore_index**=[True | False])


```python
test_series_1_nameless = pd.Series(['foo', 'bar', 'baz', 'foo'])
```


```python
test_series_1_nameless
```




    0    foo
    1    bar
    2    baz
    3    foo
    dtype: object




```python
match_strings(test_series_1_nameless)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_index</th>
      <th>left_side</th>
      <th>similarity</th>
      <th>right_side</th>
      <th>right_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>bar</td>
      <td>1.0</td>
      <td>bar</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>baz</td>
      <td>1.0</td>
      <td>baz</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
match_strings(test_series_1_nameless, ignore_index=True)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_side</th>
      <th>similarity</th>
      <th>right_side</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bar</td>
      <td>1.0</td>
      <td>bar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>baz</td>
      <td>1.0</td>
      <td>baz</td>
    </tr>
    <tr>
      <th>4</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>5</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_series_1_named = pd.Series(['foo', 'bar', 'baz', 'foo'], name='wow')
```


```python
test_series_1_named
```




    0    foo
    1    bar
    2    baz
    3    foo
    Name: wow, dtype: object




```python
match_strings(test_series_1_named)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_index</th>
      <th>left_wow</th>
      <th>similarity</th>
      <th>right_wow</th>
      <th>right_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>bar</td>
      <td>1.0</td>
      <td>bar</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>baz</td>
      <td>1.0</td>
      <td>baz</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
match_strings(test_series_1_named, ignore_index=True)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_wow</th>
      <th>similarity</th>
      <th>right_wow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bar</td>
      <td>1.0</td>
      <td>bar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>baz</td>
      <td>1.0</td>
      <td>baz</td>
    </tr>
    <tr>
      <th>4</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>5</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_series_1_nameless_index = pd.Series(['foo', 'bar', 'baz', 'foo'], name='wow', index=list('ABCD'))
```


```python
test_series_1_nameless_index
```




    A    foo
    B    bar
    C    baz
    D    foo
    Name: wow, dtype: object




```python
match_strings(test_series_1_nameless_index)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_index</th>
      <th>left_wow</th>
      <th>similarity</th>
      <th>right_wow</th>
      <th>right_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>D</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>bar</td>
      <td>1.0</td>
      <td>bar</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C</td>
      <td>baz</td>
      <td>1.0</td>
      <td>baz</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>D</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>D</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>D</td>
    </tr>
  </tbody>
</table>
</div>




```python
match_strings(test_series_1_nameless_index, ignore_index=True)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_wow</th>
      <th>similarity</th>
      <th>right_wow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bar</td>
      <td>1.0</td>
      <td>bar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>baz</td>
      <td>1.0</td>
      <td>baz</td>
    </tr>
    <tr>
      <th>4</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>5</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_series_1_named_index = pd.Series(['foo', 'bar', 'baz', 'foo'], name='wow', index=list('ABCD')).rename_axis('id')
```


```python
test_series_1_named_index
```




    id
    A    foo
    B    bar
    C    baz
    D    foo
    Name: wow, dtype: object




```python
match_strings(test_series_1_named_index)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_id</th>
      <th>left_wow</th>
      <th>similarity</th>
      <th>right_wow</th>
      <th>right_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>D</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>bar</td>
      <td>1.0</td>
      <td>bar</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C</td>
      <td>baz</td>
      <td>1.0</td>
      <td>baz</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>D</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>D</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>D</td>
    </tr>
  </tbody>
</table>
</div>




```python
match_strings(test_series_1_named_index, ignore_index=True)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_wow</th>
      <th>similarity</th>
      <th>right_wow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bar</td>
      <td>1.0</td>
      <td>bar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>baz</td>
      <td>1.0</td>
      <td>baz</td>
    </tr>
    <tr>
      <th>4</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>5</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df = pd.DataFrame(
    {
        'label': ['foo', 'bar', 'baz', 'foo'],
         'LVL1': list('ABCD'),
         'LVL2': [0, 1, 2, 3]
    }
).set_index(['LVL1', 'LVL2']).squeeze()
```


```python
test_df
```




    LVL1  LVL2
    A     0       foo
    B     1       bar
    C     2       baz
    D     3       foo
    Name: label, dtype: object




```python
match_strings(test_df)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_LVL1</th>
      <th>left_LVL2</th>
      <th>left_label</th>
      <th>similarity</th>
      <th>right_label</th>
      <th>right_LVL2</th>
      <th>right_LVL1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>0</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>3</td>
      <td>D</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>1</td>
      <td>bar</td>
      <td>1.0</td>
      <td>bar</td>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C</td>
      <td>2</td>
      <td>baz</td>
      <td>1.0</td>
      <td>baz</td>
      <td>2</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>D</td>
      <td>3</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>D</td>
      <td>3</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>3</td>
      <td>D</td>
    </tr>
  </tbody>
</table>
</div>




```python
match_strings(test_df, ignore_index=True)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_label</th>
      <th>similarity</th>
      <th>right_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bar</td>
      <td>1.0</td>
      <td>bar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>baz</td>
      <td>1.0</td>
      <td>baz</td>
    </tr>
    <tr>
      <th>4</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>5</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df2 = pd.Series(
    ['foo', 'bar', 'baz', 'foo'], 
    index=pd.MultiIndex.from_tuples(list(zip(list('ABCD'), [0, 1, 2, 3])))
)
```


```python
test_df2
```




    A  0    foo
    B  1    bar
    C  2    baz
    D  3    foo
    dtype: object




```python
match_strings(test_df2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_level_0</th>
      <th>left_level_1</th>
      <th>left_side</th>
      <th>similarity</th>
      <th>right_side</th>
      <th>right_level_1</th>
      <th>right_level_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>0</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>3</td>
      <td>D</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>1</td>
      <td>bar</td>
      <td>1.0</td>
      <td>bar</td>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C</td>
      <td>2</td>
      <td>baz</td>
      <td>1.0</td>
      <td>baz</td>
      <td>2</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>D</td>
      <td>3</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>D</td>
      <td>3</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>3</td>
      <td>D</td>
    </tr>
  </tbody>
</table>
</div>




```python
match_strings(test_df2, ignore_index=True)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_side</th>
      <th>similarity</th>
      <th>right_side</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bar</td>
      <td>1.0</td>
      <td>bar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>baz</td>
      <td>1.0</td>
      <td>baz</td>
    </tr>
    <tr>
      <th>4</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>5</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
match_strings(test_df2, test_series_1_named)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_level_0</th>
      <th>left_level_1</th>
      <th>left_side</th>
      <th>similarity</th>
      <th>right_wow</th>
      <th>right_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>0</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>1</td>
      <td>bar</td>
      <td>1.0</td>
      <td>bar</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C</td>
      <td>2</td>
      <td>baz</td>
      <td>1.0</td>
      <td>baz</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>D</td>
      <td>3</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>D</td>
      <td>3</td>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
match_strings(test_df2, test_series_1_named, ignore_index=True)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_side</th>
      <th>similarity</th>
      <th>right_wow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bar</td>
      <td>1.0</td>
      <td>bar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>baz</td>
      <td>1.0</td>
      <td>baz</td>
    </tr>
    <tr>
      <th>4</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>5</th>
      <td>foo</td>
      <td>1.0</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>



# 2. **group_similar_strings**(..., **ignore_index**=[True | False])

Let's import some data:


```python
companies_df = pd.read_csv('data/sec__edgar_company_info.csv')[0:50000]
```


```python
companies_df.squeeze()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Line Number</th>
      <th>Company Name</th>
      <th>Company CIK Key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>!J INC</td>
      <td>1438823</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>#1 A LIFESAFER HOLDINGS, INC.</td>
      <td>1509607</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
      <td>1457512</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>#1 PAINTBALL CORP</td>
      <td>1433777</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>$ LLC</td>
      <td>1427189</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49995</th>
      <td>49996</td>
      <td>BABB DOUGLAS J</td>
      <td>1190359</td>
    </tr>
    <tr>
      <th>49996</th>
      <td>49997</td>
      <td>BABB HENRY C</td>
      <td>1193948</td>
    </tr>
    <tr>
      <th>49997</th>
      <td>49998</td>
      <td>BABB INTERNATIONAL INC</td>
      <td>1139504</td>
    </tr>
    <tr>
      <th>49998</th>
      <td>49999</td>
      <td>BABB JACK J</td>
      <td>1280368</td>
    </tr>
    <tr>
      <th>49999</th>
      <td>50000</td>
      <td>BABB JAMES G. III</td>
      <td>1575424</td>
    </tr>
  </tbody>
</table>
<p>50000 rows &times; 3 columns</p>
</div>



Let's give the data a different (unique-valued) index as is commonly done:


```python
companies_df.set_index(['Line Number', 'Company CIK Key'], inplace=True, verify_integrity=True)
```


```python
companies_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Company Name</th>
    </tr>
    <tr>
      <th>Line Number</th>
      <th>Company CIK Key</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <th>1438823</th>
      <td>!J INC</td>
    </tr>
    <tr>
      <th>2</th>
      <th>1509607</th>
      <td>#1 A LIFESAFER HOLDINGS, INC.</td>
    </tr>
    <tr>
      <th>3</th>
      <th>1457512</th>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
    </tr>
    <tr>
      <th>4</th>
      <th>1433777</th>
      <td>#1 PAINTBALL CORP</td>
    </tr>
    <tr>
      <th>5</th>
      <th>1427189</th>
      <td>$ LLC</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>49996</th>
      <th>1190359</th>
      <td>BABB DOUGLAS J</td>
    </tr>
    <tr>
      <th>49997</th>
      <th>1193948</th>
      <td>BABB HENRY C</td>
    </tr>
    <tr>
      <th>49998</th>
      <th>1139504</th>
      <td>BABB INTERNATIONAL INC</td>
    </tr>
    <tr>
      <th>49999</th>
      <th>1280368</th>
      <td>BABB JACK J</td>
    </tr>
    <tr>
      <th>50000</th>
      <th>1575424</th>
      <td>BABB JAMES G. III</td>
    </tr>
  </tbody>
</table>
<p>50000 rows &times; 1 columns</p>
</div>



Now let's do some grouping as usual:


```python
companies = companies_df.copy()
```


```python
group_similar_strings(companies['Company Name'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>group_rep_Line Number</th>
      <th>group_rep_Company CIK Key</th>
      <th>group_rep</th>
    </tr>
    <tr>
      <th>Line Number</th>
      <th>Company CIK Key</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <th>1438823</th>
      <td>1</td>
      <td>1438823</td>
      <td>!J INC</td>
    </tr>
    <tr>
      <th>2</th>
      <th>1509607</th>
      <td>2</td>
      <td>1509607</td>
      <td>#1 A LIFESAFER HOLDINGS, INC.</td>
    </tr>
    <tr>
      <th>3</th>
      <th>1457512</th>
      <td>3</td>
      <td>1457512</td>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
    </tr>
    <tr>
      <th>4</th>
      <th>1433777</th>
      <td>4</td>
      <td>1433777</td>
      <td>#1 PAINTBALL CORP</td>
    </tr>
    <tr>
      <th>5</th>
      <th>1427189</th>
      <td>5</td>
      <td>1427189</td>
      <td>$ LLC</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49996</th>
      <th>1190359</th>
      <td>49996</td>
      <td>1190359</td>
      <td>BABB DOUGLAS J</td>
    </tr>
    <tr>
      <th>49997</th>
      <th>1193948</th>
      <td>49997</td>
      <td>1193948</td>
      <td>BABB HENRY C</td>
    </tr>
    <tr>
      <th>49998</th>
      <th>1139504</th>
      <td>49998</td>
      <td>1139504</td>
      <td>BABB INTERNATIONAL INC</td>
    </tr>
    <tr>
      <th>49999</th>
      <th>1280368</th>
      <td>49999</td>
      <td>1280368</td>
      <td>BABB JACK J</td>
    </tr>
    <tr>
      <th>50000</th>
      <th>1575424</th>
      <td>50000</td>
      <td>1575424</td>
      <td>BABB JAMES G. III</td>
    </tr>
  </tbody>
</table>
<p>50000 rows &times; 3 columns</p>
</div>



Notice that `group_similar_strings` preeserves the index of the input Series while also showing the index(es) of the group-representatives in new columns with column-names prefixed by the string "group_rep_". 

To ignore the indexes of the group-representatives, simply set the keyword argument `ignore_index = True`:


```python
group_similar_strings(companies['Company Name'], ignore_index=True)
```




    Line Number  Company CIK Key
    1            1438823                                        !J INC
    2            1509607                 #1 A LIFESAFER HOLDINGS, INC.
    3            1457512            #1 ARIZONA DISCOUNT PROPERTIES LLC
    4            1433777                             #1 PAINTBALL CORP
    5            1427189                                         $ LLC
                                                   ...                
    49996        1190359                                BABB DOUGLAS J
    49997        1193948                                  BABB HENRY C
    49998        1139504                        BABB INTERNATIONAL INC
    49999        1280368                                   BABB JACK J
    50000        1575424                             BABB JAMES G. III
    Name: group_rep, Length: 50000, dtype: object



Because the output always inherits the index of the input Series, it is possible to directly assign it to new columns of the `companies` DataFrame (which has the exact same index) while also giving them new column names, as in the following:


```python
companies[['Group Line Number', 'Group CIK Key', 'Group']] = \
group_similar_strings(companies['Company Name'], min_similarity=0.70)
```


```python
companies
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Company Name</th>
      <th>Group Line Number</th>
      <th>Group CIK Key</th>
      <th>Group</th>
    </tr>
    <tr>
      <th>Line Number</th>
      <th>Company CIK Key</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <th>1438823</th>
      <td>!J INC</td>
      <td>1</td>
      <td>1438823</td>
      <td>!J INC</td>
    </tr>
    <tr>
      <th>2</th>
      <th>1509607</th>
      <td>#1 A LIFESAFER HOLDINGS, INC.</td>
      <td>2</td>
      <td>1509607</td>
      <td>#1 A LIFESAFER HOLDINGS, INC.</td>
    </tr>
    <tr>
      <th>3</th>
      <th>1457512</th>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
      <td>3</td>
      <td>1457512</td>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
    </tr>
    <tr>
      <th>4</th>
      <th>1433777</th>
      <td>#1 PAINTBALL CORP</td>
      <td>4</td>
      <td>1433777</td>
      <td>#1 PAINTBALL CORP</td>
    </tr>
    <tr>
      <th>5</th>
      <th>1427189</th>
      <td>$ LLC</td>
      <td>5</td>
      <td>1427189</td>
      <td>$ LLC</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49996</th>
      <th>1190359</th>
      <td>BABB DOUGLAS J</td>
      <td>49996</td>
      <td>1190359</td>
      <td>BABB DOUGLAS J</td>
    </tr>
    <tr>
      <th>49997</th>
      <th>1193948</th>
      <td>BABB HENRY C</td>
      <td>49997</td>
      <td>1193948</td>
      <td>BABB HENRY C</td>
    </tr>
    <tr>
      <th>49998</th>
      <th>1139504</th>
      <td>BABB INTERNATIONAL INC</td>
      <td>49998</td>
      <td>1139504</td>
      <td>BABB INTERNATIONAL INC</td>
    </tr>
    <tr>
      <th>49999</th>
      <th>1280368</th>
      <td>BABB JACK J</td>
      <td>49999</td>
      <td>1280368</td>
      <td>BABB JACK J</td>
    </tr>
    <tr>
      <th>50000</th>
      <th>1575424</th>
      <td>BABB JAMES G. III</td>
      <td>50000</td>
      <td>1575424</td>
      <td>BABB JAMES G. III</td>
    </tr>
  </tbody>
</table>
<p>50000 rows &times; 4 columns</p>
</div>



The grouping is not readily seen in the above display.  So let us determine the number of members of each group (the group size) and then sort by group size:


```python
companies['Group Size'] = \
companies\
.groupby(['Group Line Number', 'Group CIK Key'], as_index=False)['Company Name']\
.transform('count')
companies.sort_values('Group Size', ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Company Name</th>
      <th>Group Line Number</th>
      <th>Group CIK Key</th>
      <th>Group</th>
      <th>Group Size</th>
    </tr>
    <tr>
      <th>Line Number</th>
      <th>Company CIK Key</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14060</th>
      <th>1425318</th>
      <td>ADVISORS DISCIPLINED TRUST 241</td>
      <td>14940</td>
      <td>1297377</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>1866</td>
    </tr>
    <tr>
      <th>13845</th>
      <th>1662291</th>
      <td>ADVISORS DISCIPLINED TRUST 1674</td>
      <td>14940</td>
      <td>1297377</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>1866</td>
    </tr>
    <tr>
      <th>13821</th>
      <th>1662315</th>
      <td>ADVISORS DISCIPLINED TRUST 1652</td>
      <td>14940</td>
      <td>1297377</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>1866</td>
    </tr>
    <tr>
      <th>13822</th>
      <th>1662313</th>
      <td>ADVISORS DISCIPLINED TRUST 1653</td>
      <td>14940</td>
      <td>1297377</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>1866</td>
    </tr>
    <tr>
      <th>13823</th>
      <th>1662312</th>
      <td>ADVISORS DISCIPLINED TRUST 1654</td>
      <td>14940</td>
      <td>1297377</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>1866</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20536</th>
      <th>1529695</th>
      <td>ALENTUS CORP</td>
      <td>20536</td>
      <td>1529695</td>
      <td>ALENTUS CORP</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20535</th>
      <th>1585882</th>
      <td>ALENT PLC/ADR</td>
      <td>20535</td>
      <td>1585882</td>
      <td>ALENT PLC/ADR</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20534</th>
      <th>1688865</th>
      <td>ALENSON CARMAN</td>
      <td>20534</td>
      <td>1688865</td>
      <td>ALENSON CARMAN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20533</th>
      <th>3433</th>
      <td>ALENICK JEROME B</td>
      <td>20533</td>
      <td>3433</td>
      <td>ALENICK JEROME B</td>
      <td>1</td>
    </tr>
    <tr>
      <th>50000</th>
      <th>1575424</th>
      <td>BABB JAMES G. III</td>
      <td>50000</td>
      <td>1575424</td>
      <td>BABB JAMES G. III</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>50000 rows &times; 5 columns</p>
</div>



Let's see where the largest group 'ADVISORS DISCIPLINED TRUST' ends:


```python
companies.sort_values('Group Size', ascending=False).iloc[(start:=1861):(start + 10)]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Company Name</th>
      <th>Group Line Number</th>
      <th>Group CIK Key</th>
      <th>Group</th>
      <th>Group Size</th>
    </tr>
    <tr>
      <th>Line Number</th>
      <th>Company CIK Key</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13248</th>
      <th>1578605</th>
      <td>ADVISORS DISCIPLINED TRUST 1131</td>
      <td>14940</td>
      <td>1297377</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>1866</td>
    </tr>
    <tr>
      <th>13265</th>
      <th>1582225</th>
      <td>ADVISORS DISCIPLINED TRUST 1147</td>
      <td>14940</td>
      <td>1297377</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>1866</td>
    </tr>
    <tr>
      <th>13250</th>
      <th>1578617</th>
      <td>ADVISORS DISCIPLINED TRUST 1133</td>
      <td>14940</td>
      <td>1297377</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>1866</td>
    </tr>
    <tr>
      <th>13249</th>
      <th>1578604</th>
      <td>ADVISORS DISCIPLINED TRUST 1132</td>
      <td>14940</td>
      <td>1297377</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>1866</td>
    </tr>
    <tr>
      <th>13264</th>
      <th>1582226</th>
      <td>ADVISORS DISCIPLINED TRUST 1146</td>
      <td>14940</td>
      <td>1297377</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>1866</td>
    </tr>
    <tr>
      <th>32937</th>
      <th>1643285</th>
      <td>ANGELLIST-SSTY-FUND, A SERIES OF ANGELLIST-FG-...</td>
      <td>32535</td>
      <td>1600532</td>
      <td>ANGELLIST-ART-FUND, A SERIES OF ANGELLIST-FG-F...</td>
      <td>485</td>
    </tr>
    <tr>
      <th>32552</th>
      <th>1677061</th>
      <td>ANGELLIST-BDID-PR-FUND, A SERIES OF ANGELLIST-...</td>
      <td>32535</td>
      <td>1600532</td>
      <td>ANGELLIST-ART-FUND, A SERIES OF ANGELLIST-FG-F...</td>
      <td>485</td>
    </tr>
    <tr>
      <th>32943</th>
      <th>1623545</th>
      <td>ANGELLIST-STHE-FUND, A SERIES OF ANGELLIST-GP-...</td>
      <td>32535</td>
      <td>1600532</td>
      <td>ANGELLIST-ART-FUND, A SERIES OF ANGELLIST-FG-F...</td>
      <td>485</td>
    </tr>
    <tr>
      <th>32944</th>
      <th>1680681</th>
      <td>ANGELLIST-STHE-PR-FUND, A SERIES OF ANGELLIST-...</td>
      <td>32535</td>
      <td>1600532</td>
      <td>ANGELLIST-ART-FUND, A SERIES OF ANGELLIST-FG-F...</td>
      <td>485</td>
    </tr>
    <tr>
      <th>32947</th>
      <th>1610185</th>
      <td>ANGELLIST-SUIT-FUND, A SERIES OF ANGELLIST-FGR...</td>
      <td>32535</td>
      <td>1600532</td>
      <td>ANGELLIST-ART-FUND, A SERIES OF ANGELLIST-FG-F...</td>
      <td>485</td>
    </tr>
  </tbody>
</table>
</div>


# <a name="MMS"></a>3. **match_most_similar**(..., **ignore_index**=[True | False], **replace_na**=[True | False])

Now let's create a 'master' Series of group-representatives (we will use it later in the function `match_most_similar`):


```python
master = companies.groupby(['Group Line Number', 'Group CIK Key'])['Group'].first().head(-5)
```

Notice that we have intentionally excluded the last five groups.


```python
master.index.rename(['Line Number', 'Company CIK Key'], inplace=True)
master.rename('Company Name', inplace=True)
```




    Line Number  Company CIK Key
    1            1438823                                        !J INC
    2            1509607                 #1 A LIFESAFER HOLDINGS, INC.
    3            1457512            #1 ARIZONA DISCOUNT PROPERTIES LLC
    4            1433777                             #1 PAINTBALL CORP
    5            1427189                                         $ LLC
                                                   ...                
    49991        1615648                                BABACAN THOMAS
    49992        1443093                            BABAD SHOLOM CHAIM
    49993        1208255                                    BABALU LLC
    49994        1270229                                  BABANI SUSIE
    49995        1660243                                   BABAY KARIM
    Name: Company Name, Length: 34249, dtype: object



Let's examine the neighbourhood of the largest group 'ADVISORS DISCIPLINED TRUST':


```python
master.iloc[(start:=(master.index.get_loc((14940, 1297377)) - 5)):(start + 10)]
```




    Line Number  Company CIK Key
    13087        1029068                            ADVISORONE FUNDS
    13089        1313535                           ADVISORPORT, INC.
    13090        1559198                                ADVISORS 999
    13094        789623             ADVISORS CAPITAL INVESTMENTS INC
    13096        932536              ADVISORS CLEARING NETWORK, INC.
    14940        1297377                  ADVISORS DISCIPLINED TRUST
    14954        1313525               ADVISORS EDGE SECURITIES, LLC
    14955        825201                            ADVISORS FUND L P
    14957        1267654                         ADVISORS GENPAR INC
    14958        1016084                      ADVISORS GROUP INC /DC
    Name: Company Name, dtype: object



Now let's use `master` in function `match_most_similar`:


```python
companies = companies_df.copy()
```


```python
grouped_data = match_most_similar(
    master,
    companies['Company Name'],
    min_similarity=0.55,
    max_n_matches=2000
)

grouped_data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>most_similar_Line Number</th>
      <th>most_similar_Company CIK Key</th>
      <th>most_similar_Company Name</th>
    </tr>
    <tr>
      <th>Line Number</th>
      <th>Company CIK Key</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <th>1438823</th>
      <td>1.0</td>
      <td>1438823.0</td>
      <td>!J INC</td>
    </tr>
    <tr>
      <th>2</th>
      <th>1509607</th>
      <td>2.0</td>
      <td>1509607.0</td>
      <td>#1 A LIFESAFER HOLDINGS, INC.</td>
    </tr>
    <tr>
      <th>3</th>
      <th>1457512</th>
      <td>3.0</td>
      <td>1457512.0</td>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
    </tr>
    <tr>
      <th>4</th>
      <th>1433777</th>
      <td>4.0</td>
      <td>1433777.0</td>
      <td>#1 PAINTBALL CORP</td>
    </tr>
    <tr>
      <th>5</th>
      <th>1427189</th>
      <td>5.0</td>
      <td>1427189.0</td>
      <td>$ LLC</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49996</th>
      <th>1190359</th>
      <td>31995.0</td>
      <td>1336287.0</td>
      <td>ANDREA DOUGLAS J</td>
    </tr>
    <tr>
      <th>49997</th>
      <th>1193948</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>BABB HENRY C</td>
    </tr>
    <tr>
      <th>49998</th>
      <th>1139504</th>
      <td>19399.0</td>
      <td>1569329.0</td>
      <td>AL INTERNATIONAL, INC.</td>
    </tr>
    <tr>
      <th>49999</th>
      <th>1280368</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>BABB JACK J</td>
    </tr>
    <tr>
      <th>50000</th>
      <th>1575424</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>BABB JAMES G. III</td>
    </tr>
  </tbody>
</table>
<p>50000 rows &times; 3 columns</p>
</div>



Notice that `match_most_similar` also preserves the indexes of the input `duplicates` Series.  

Also, notice that the indexes of the matched strings (including the matched strings themselves) in `master` appear in the output as columns whose names are prefixed with the string "most_similar_".

Finally, notice the 'NaN' values in the index columns corresponding to those strings in `duplicates` that have no match in `master` (above the similarity threshold specified by the keyword argument setting `min_similarity=0.55`).  Recall that we earlier intentionally excluded certain groups in `master`.  These 'NaN' values are a consequence of that exclusion. _Apparently, it is also because of these 'NaN' values that these index columns have been converted into float data-types._

We can replace the 'NaN' values with their corresponding index values by setting the keyword argument `replace_na=True`. (The reason why the function `match_most_similar` does not do this by default is because in general `duplicates` may have a different index from `master` with possibly a different number of index levels.)


```python
grouped_data = match_most_similar(
    master,
    companies['Company Name'],
    min_similarity=0.55,
    max_n_matches=2000,
    replace_na=True
)

grouped_data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>most_similar_Line Number</th>
      <th>most_similar_Company CIK Key</th>
      <th>most_similar_Company Name</th>
    </tr>
    <tr>
      <th>Line Number</th>
      <th>Company CIK Key</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <th>1438823</th>
      <td>1</td>
      <td>1438823</td>
      <td>!J INC</td>
    </tr>
    <tr>
      <th>2</th>
      <th>1509607</th>
      <td>2</td>
      <td>1509607</td>
      <td>#1 A LIFESAFER HOLDINGS, INC.</td>
    </tr>
    <tr>
      <th>3</th>
      <th>1457512</th>
      <td>3</td>
      <td>1457512</td>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
    </tr>
    <tr>
      <th>4</th>
      <th>1433777</th>
      <td>4</td>
      <td>1433777</td>
      <td>#1 PAINTBALL CORP</td>
    </tr>
    <tr>
      <th>5</th>
      <th>1427189</th>
      <td>5</td>
      <td>1427189</td>
      <td>$ LLC</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49996</th>
      <th>1190359</th>
      <td>31995</td>
      <td>1336287</td>
      <td>ANDREA DOUGLAS J</td>
    </tr>
    <tr>
      <th>49997</th>
      <th>1193948</th>
      <td>49997</td>
      <td>1193948</td>
      <td>BABB HENRY C</td>
    </tr>
    <tr>
      <th>49998</th>
      <th>1139504</th>
      <td>19399</td>
      <td>1569329</td>
      <td>AL INTERNATIONAL, INC.</td>
    </tr>
    <tr>
      <th>49999</th>
      <th>1280368</th>
      <td>49999</td>
      <td>1280368</td>
      <td>BABB JACK J</td>
    </tr>
    <tr>
      <th>50000</th>
      <th>1575424</th>
      <td>50000</td>
      <td>1575424</td>
      <td>BABB JAMES G. III</td>
    </tr>
  </tbody>
</table>
<p>50000 rows &times; 3 columns</p>
</div>



Let's inspect the result by determining the group sizes:


```python
grouped_data.groupby('most_similar_Company Name')['most_similar_Line Number'].count()\
.rename('Size')\
.sort_values(ascending=False)
```




    most_similar_Company Name
    ADVISORS DISCIPLINED TRUST                                 1866
    ANGELLIST-ART-FUND, A SERIES OF ANGELLIST-FG-FUNDS, LLC     279
    ALTERNATIVE LOAN TRUST 2005-4                               193
    AGL LIFE ASSURANCE CO SEPARATE ACCOUNT                      188
    AMERICREDIT AUTOMOBILE RECEIVABLES TRUST 2001-1              89
                                                               ... 
    AIM SAFETY CO INC                                             1
    AIM REAL ESTATE HEDGED EQUITY (U.S.) FUND, LP                 1
    AIM QUANTITATIVE GLOBAL SF LP                                 1
    AIM OXFORD HOLDINGS, LLC                                      1
    TALISMAN ENERGY SWEDEN AB                                     1
    Name: Size, Length: 34496, dtype: int64



Just like we did for function `group_similar_strings`, we can ignore the indexes of `master` if we choose, by setting `ignore_index=True`:


```python
grouped_data_dropped = match_most_similar(
    master,
    companies['Company Name'],
    ignore_index=True,
    min_similarity=0.55,
    max_n_matches=2000
)
```


```python
grouped_data_dropped
```




    Line Number  Company CIK Key
    1            1438823                                        !J INC
    2            1509607                 #1 A LIFESAFER HOLDINGS, INC.
    3            1457512            #1 ARIZONA DISCOUNT PROPERTIES LLC
    4            1433777                             #1 PAINTBALL CORP
    5            1427189                                         $ LLC
                                                   ...                
    49996        1190359                              ANDREA DOUGLAS J
    49997        1193948                                  BABB HENRY C
    49998        1139504                        AL INTERNATIONAL, INC.
    49999        1280368                                   BABB JACK J
    50000        1575424                             BABB JAMES G. III
    Name: most_similar_Company Name, Length: 50000, dtype: object



As before, we can here also directly assign the output to new columns of the `companies` DataFrame (because it has the exact same index) while also giving them new column names, as in the following:


```python
companies['Most Similar Name'] = grouped_data_dropped
```


```python
companies
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Company Name</th>
      <th>Most Similar Name</th>
    </tr>
    <tr>
      <th>Line Number</th>
      <th>Company CIK Key</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <th>1438823</th>
      <td>!J INC</td>
      <td>!J INC</td>
    </tr>
    <tr>
      <th>2</th>
      <th>1509607</th>
      <td>#1 A LIFESAFER HOLDINGS, INC.</td>
      <td>#1 A LIFESAFER HOLDINGS, INC.</td>
    </tr>
    <tr>
      <th>3</th>
      <th>1457512</th>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
    </tr>
    <tr>
      <th>4</th>
      <th>1433777</th>
      <td>#1 PAINTBALL CORP</td>
      <td>#1 PAINTBALL CORP</td>
    </tr>
    <tr>
      <th>5</th>
      <th>1427189</th>
      <td>$ LLC</td>
      <td>$ LLC</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49996</th>
      <th>1190359</th>
      <td>BABB DOUGLAS J</td>
      <td>ANDREA DOUGLAS J</td>
    </tr>
    <tr>
      <th>49997</th>
      <th>1193948</th>
      <td>BABB HENRY C</td>
      <td>BABB HENRY C</td>
    </tr>
    <tr>
      <th>49998</th>
      <th>1139504</th>
      <td>BABB INTERNATIONAL INC</td>
      <td>AL INTERNATIONAL, INC.</td>
    </tr>
    <tr>
      <th>49999</th>
      <th>1280368</th>
      <td>BABB JACK J</td>
      <td>BABB JACK J</td>
    </tr>
    <tr>
      <th>50000</th>
      <th>1575424</th>
      <td>BABB JAMES G. III</td>
      <td>BABB JAMES G. III</td>
    </tr>
  </tbody>
</table>
<p>50000 rows &times; 2 columns</p>
</div>


# <a name="SG"></a>4. **StringGrouper**(..., **ignore_index**=[True | False], **replace_na**=[True | False])
The options `ignore_index` and `replace_na` can be passed directly to  a `StringGrouper` object during instantiation.  These will be used by its methods `StringGrouper.get_groups` and `StringGrouper.get_matches`.
 
The options `ignore_index` and `replace_na` can also, where applicable, be passed directly to `StringGrouper.get_groups` and `StringGrouper.get_matches` themselves to temporarily override the StringGrouper-instance's defaults. 


