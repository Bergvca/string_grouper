# When min_similarity &le; 0 and include_zeroes = [True | False])


```python
import pandas as pd
import numpy as np
from string_grouper import StringGrouper
```


```python
companies_df = pd.read_csv('data/sec__edgar_company_info.csv')[0:50000]
```


```python
master = companies_df['Company Name']
master_id = companies_df['Line Number']
duplicates = pd.Series(["ADVISORS DISCIPLINED TRUST", "ADVISORS DISCIPLINED TRUST '18"])
duplicates_id = pd.Series([3, 5])
```

#### When ID's are passed as arguments:
By default, zero-similarity matches are found and output when `min_similarity = 0`:


```python
string_grouper = StringGrouper(
    master = master,
    duplicates=duplicates,
    master_id=master_id,
    duplicates_id=duplicates_id,
    ignore_index=True,
    min_similarity = 0,
    max_n_matches = 10000,
    regex = "[,-./#]"
).fit()
string_grouper.get_matches()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_Company Name</th>
      <th>left_Line Number</th>
      <th>similarity</th>
      <th>right_id</th>
      <th>right_side</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
      <td>3</td>
      <td>0.091157</td>
      <td>3</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
      <td>3</td>
      <td>0.063861</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>05 CAT THIEF/GOLD IN MY STARS LLC</td>
      <td>21</td>
      <td>0.015313</td>
      <td>3</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
    </tr>
    <tr>
      <th>3</th>
      <td>05 CAT THIEF/GOLD IN MY STARS LLC</td>
      <td>21</td>
      <td>0.010728</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>05 DIXIE UNION/UNDER FIRE LLC</td>
      <td>22</td>
      <td>0.025397</td>
      <td>3</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>ALLDREDGE WILLIAM T</td>
      <td>21746</td>
      <td>0.000000</td>
      <td>3</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>ALLEN SAMUEL R</td>
      <td>22183</td>
      <td>0.000000</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>ATSP INNOVATIONS, LLC</td>
      <td>45273</td>
      <td>0.000000</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>ATLAS IDF, LP</td>
      <td>44877</td>
      <td>0.000000</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>AU LEO Y</td>
      <td>45535</td>
      <td>0.000000</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
  </tbody>
</table>
<p>100000 rows &times; 5 columns</p>
</div>



#### `StringGrouper` also includes option `include_zeroes`:


```python
string_grouper = StringGrouper(
    master = master,
    duplicates=duplicates,
    master_id=master_id,
    duplicates_id=duplicates_id,
    ignore_index=True,
    min_similarity = 0,
    max_n_matches = 10000,
    regex = "[,-./#]",
    include_zeroes = False
).fit()
string_grouper.get_matches()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_Company Name</th>
      <th>left_Line Number</th>
      <th>similarity</th>
      <th>right_id</th>
      <th>right_side</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
      <td>3</td>
      <td>0.091157</td>
      <td>3</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
      <td>3</td>
      <td>0.063861</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>05 CAT THIEF/GOLD IN MY STARS LLC</td>
      <td>21</td>
      <td>0.015313</td>
      <td>3</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
    </tr>
    <tr>
      <th>3</th>
      <td>05 CAT THIEF/GOLD IN MY STARS LLC</td>
      <td>21</td>
      <td>0.010728</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>05 DIXIE UNION/UNDER FIRE LLC</td>
      <td>22</td>
      <td>0.025397</td>
      <td>3</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>28754</th>
      <td>BAAPLIFE3-2015, LLC</td>
      <td>49976</td>
      <td>0.021830</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
    <tr>
      <th>28755</th>
      <td>BAAPLIFE4-2016, LLC</td>
      <td>49977</td>
      <td>0.030983</td>
      <td>3</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
    </tr>
    <tr>
      <th>28756</th>
      <td>BAAPLIFE4-2016, LLC</td>
      <td>49977</td>
      <td>0.021706</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
    <tr>
      <th>28757</th>
      <td>BABA JOE DIAMOND VENTURES US INC.</td>
      <td>49989</td>
      <td>0.027064</td>
      <td>3</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
    </tr>
    <tr>
      <th>28758</th>
      <td>BABA JOE DIAMOND VENTURES US INC.</td>
      <td>49989</td>
      <td>0.018960</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
  </tbody>
</table>
<p>28759 rows &times; 5 columns</p>
</div>



#### `get_matches` option `include_zeroes` can override `StringGrouper` default:


```python
string_grouper.get_matches(include_zeroes=True)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_Company Name</th>
      <th>left_Line Number</th>
      <th>similarity</th>
      <th>right_id</th>
      <th>right_side</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
      <td>3</td>
      <td>0.091157</td>
      <td>3</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
      <td>3</td>
      <td>0.063861</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>05 CAT THIEF/GOLD IN MY STARS LLC</td>
      <td>21</td>
      <td>0.015313</td>
      <td>3</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
    </tr>
    <tr>
      <th>3</th>
      <td>05 CAT THIEF/GOLD IN MY STARS LLC</td>
      <td>21</td>
      <td>0.010728</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>05 DIXIE UNION/UNDER FIRE LLC</td>
      <td>22</td>
      <td>0.025397</td>
      <td>3</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>ALLDREDGE WILLIAM T</td>
      <td>21746</td>
      <td>0.000000</td>
      <td>3</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>ALLEN SAMUEL R</td>
      <td>22183</td>
      <td>0.000000</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>ATSP INNOVATIONS, LLC</td>
      <td>45273</td>
      <td>0.000000</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>ATLAS IDF, LP</td>
      <td>44877</td>
      <td>0.000000</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>AU LEO Y</td>
      <td>45535</td>
      <td>0.000000</td>
      <td>5</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
    </tr>
  </tbody>
</table>
<p>100000 rows &times; 5 columns</p>
</div>



#### When no ID's are passed as arguments and indexes are not set:
Default indexes are output:


```python
string_grouper = StringGrouper(
    master = master,
    duplicates=duplicates,
    min_similarity = 0,
    max_n_matches = 10000,
    regex = "[,-./#]"
).fit()
string_grouper.get_matches()
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
      <td>2</td>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
      <td>0.091157</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
      <td>0.063861</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>05 CAT THIEF/GOLD IN MY STARS LLC</td>
      <td>0.015313</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>05 CAT THIEF/GOLD IN MY STARS LLC</td>
      <td>0.010728</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21</td>
      <td>05 DIXIE UNION/UNDER FIRE LLC</td>
      <td>0.025397</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>21745</td>
      <td>ALLDREDGE WILLIAM T</td>
      <td>0.000000</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>22182</td>
      <td>ALLEN SAMUEL R</td>
      <td>0.000000</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>45272</td>
      <td>ATSP INNOVATIONS, LLC</td>
      <td>0.000000</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>44876</td>
      <td>ATLAS IDF, LP</td>
      <td>0.000000</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>45534</td>
      <td>AU LEO Y</td>
      <td>0.000000</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>100000 rows &times; 5 columns</p>
</div>



#### When no ID's are passed as arguments but indexes are set:
Indexes are output:


```python
master.index = pd.Index(master_id)
duplicates.index = pd.Index(duplicates_id)
string_grouper = StringGrouper(
    master = master,
    duplicates=duplicates,
    min_similarity = 0,
    max_n_matches = 10000,
    regex = "[,-./#]"
).fit()
string_grouper.get_matches()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_Line Number</th>
      <th>left_Company Name</th>
      <th>similarity</th>
      <th>right_side</th>
      <th>right_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
      <td>0.091157</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
      <td>0.063861</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>05 CAT THIEF/GOLD IN MY STARS LLC</td>
      <td>0.015313</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>05 CAT THIEF/GOLD IN MY STARS LLC</td>
      <td>0.010728</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>05 DIXIE UNION/UNDER FIRE LLC</td>
      <td>0.025397</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>21746</td>
      <td>ALLDREDGE WILLIAM T</td>
      <td>0.000000</td>
      <td>ADVISORS DISCIPLINED TRUST</td>
      <td>3</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>22183</td>
      <td>ALLEN SAMUEL R</td>
      <td>0.000000</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
      <td>5</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>45273</td>
      <td>ATSP INNOVATIONS, LLC</td>
      <td>0.000000</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
      <td>5</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>44877</td>
      <td>ATLAS IDF, LP</td>
      <td>0.000000</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
      <td>5</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>45535</td>
      <td>AU LEO Y</td>
      <td>0.000000</td>
      <td>ADVISORS DISCIPLINED TRUST '18</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>100000 rows &times; 5 columns</p>
</div>


