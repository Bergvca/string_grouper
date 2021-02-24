# Finding Duplicates With IDs In String Grouper

## Introduction

A common requirement in data clean-up is the scenario where a data set (database, pandas DataFrame) has multiple database records for the same entity and duplicates need to be found. This example will not cover the task of merging or removing duplicate records — what it will do is use String Grouper to find duplicate records using the match_strings function and the optional IDs functionality.

For the example we will use [this](accounts.csv) simple data set. The number of rows is not important, the 'name' column has a number of typical cases of types of variations in spelling.

```
id,name
AA012345X,mega enterprises corp.
BB016741P,mega enterprises corporation
CC052345T,mega corp.
AA098762D,hyper startup inc.
BB099931J,hyper-startup inc.
CC082744L,hyper startup incorporated
HH072982K,hyper hyper inc.
AA903844B,slow and steady inc.
BB904941H,slow and steady incorporated
CC903844B,slow steady inc.
AA777431C,abc enterprises inc.
BB760431Y,a.b.c. enterprises incorporated
BB750431M,a.b.c. enterprises inc.
ZZ123456H,one and only inc.
```

## Example

The steps below will process the above sample file using String Grouper to search for matches in the values in the 'name' column. The results shown in the tables at each step are based on the sample data above.

### Setup

```python
import pandas as pd
from string_grouper import match_strings
```

### Import Data

***Tip:*** Assuming the data set will come from an external database, for optimum performance only do an export of the ID column, and the text column that matching will be done on, and convert the text data column (**not the ID column**) to lower case.

#### Import the sample data.

```python
accounts = pd.read_csv('string_grouper/tutorials/accounts.csv')
# Show dataframe
accounts
```

#### Result (first three rows only shown):

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AA012345X</td>
      <td>mega enterprises corp.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BB016741P</td>
      <td>mega enterprises corporation</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CC052345T</td>
      <td>mega corp.</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
</div>


### Find matches, assign to new pandas variable

Next, use the `match_strings` function and pass the 'name' column as the argument to the `master` parameter, and the 'id' column as the argument to the `master_id` parameter.

**N.B.** In production with a real data set, depending on its size, the following command can/may take a number of minutes — ***no update/progress indicator is shown***. This obviously also depends on the performance of the computer used. Memory and hard disk performance are a factor, as well as the CPU. String Grouper uses pandas which, in turn, uses NumPy, so matching is not done by computationally intensive looping, but by [array mathematics](https://realpython.com/numpy-array-programming/) — but it still may take some time to process large data sets.

```python
matches = match_strings(accounts['name'], master_id = accounts['id'])
matches
```
This will return a pandas DataFrame as below. The values (company) we will focus on in this example will be those that have variations in the name of the fictitious company, 'Hyper Startup Inc.'.


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_side_id</th>
      <th>left_side</th>
      <th>right_side_id</th>
      <th>right_side</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AA098762D</td>
      <td>hyper startup inc.</td>
      <td>BB099931J</td>
      <td>hyper-startup inc.</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AA098762D</td>
      <td>hyper startup inc.</td>
      <td>AA098762D</td>
      <td>hyper startup inc.</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BB099931J</td>
      <td>hyper-startup inc.</td>
      <td>BB099931J</td>
      <td>hyper-startup inc.</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BB099931J</td>
      <td>hyper-startup inc.</td>
      <td>AA098762D</td>
      <td>hyper startup inc.</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CC082744L</td>
      <td>hyper startup incorporated</td>
      <td>CC082744L</td>
      <td>hyper startup incorporated</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
</div>


In a pattern-matching process, each value in a row of the column being matched is checked against *every other value* in the column. 

Processing this using typical Python looping code would mean, in the case of a 100,000 row data set, that the total iterations would be 100,000<sup>2</sup> = 10 Billion. Processing that number of iterations might require replacing the CPU of the computer after each investigation! Well maybe not ... but you *would* have time for a few cups of coffee. String Grouper works in a totally different way.

In the resultant DataFrame above, we see the IDs (AA098762D, BB099931J) having each a group of two values — once where a close match is found, and once where its own record (value) is found. The third ID, CC082744L, is only returned once, even though it is pretty clear that it would be a variation of our fictitious company 'Hyper Startup Inc.'


### Using the 'Minimum Similarity' keyword argument

String Grouper has a number of configuration options (see the **kwargs** in README.md). The option of interest in the above case is `min_similarity`. 

The default minimum similarity is 0.8. It can be seen that more matches may be found by reducing the minimum similarity from 0.8 to, for example, 0.7.

```python
matches = match_strings(accounts['name'], master_id = accounts['id'], min_similarity = 0.7)
```

***Tip:*** If the data set being matched is large, and you wish to experiment with the minimum similarity option, it may be helpful to import only a limited data set during testing, and increase to the full data set when ready. The number of rows imported can be specified in this way:

```python
# We only look at the first 50k as an example
accounts = pd.read_csv('/path/to/folder/huge_file.csv')[0:50000]
```

Back to our example ... changing the option to `min_similarity = 0.7` returns this:

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_side_id</th>
      <th>left_side</th>
      <th>right_side_id</th>
      <th>right_side</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AA098762D</td>
      <td>hyper startup inc.</td>
      <td>BB099931J</td>
      <td>hyper-startup inc.</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AA098762D</td>
      <td>hyper startup inc.</td>
      <td>AA098762D</td>
      <td>hyper startup inc.</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AA098762D</td>
      <td>hyper startup inc.</td>
      <td>CC082744L</td>
      <td>hyper startup incorporated</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BB099931J</td>
      <td>hyper-startup inc.</td>
      <td>BB099931J</td>
      <td>hyper-startup inc.</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BB099931J</td>
      <td>hyper-startup inc.</td>
      <td>AA098762D</td>
      <td>hyper startup inc.</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BB099931J</td>
      <td>hyper-startup inc.</td>
      <td>CC082744L</td>
      <td>hyper startup incorporated</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CC082744L</td>
      <td>hyper startup incorporated</td>
      <td>CC082744L</td>
      <td>hyper startup incorporated</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CC082744L</td>
      <td>hyper startup incorporated</td>
      <td>BB099931J</td>
      <td>hyper-startup inc.</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CC082744L</td>
      <td>hyper startup incorporated</td>
      <td>AA098762D</td>
      <td>hyper startup inc.</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>14</th>
      <td>HH072982K</td>
      <td>hyper hyper inc.</td>
      <td>HH072982K</td>
      <td>hyper hyper inc.</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
</div>

Now we see the IDs — AA098762D, BB099931J, CC082744L — have further matches.  Each 'name' value has two other matching rows (IDs). However, we see that setting minimum similarity to 0.7 has still not matched 'hyper hyper inc.' (ID HH072982K) even though a person would judge that the 'name' is a match. The minimum similarity setting can be adjusted up and down until it is considered that most duplicates are being matched. If so, we can progress.

### Removing identical rows

Once we are happy with the level of matching, we can remove the rows where the IDs are the same. Having the original (database) IDs for the rows means that we can precisely remove identical rows — that is, we are not removing matches based on similar values, but on the exact (database) IDs:

```python
dupes = matches[matches.left_side_id != matches.right_side_id]
dupes
```
And we see the following for the company name we have been following:


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_side_id</th>
      <th>left_side</th>
      <th>right_side_id</th>
      <th>right_side</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AA098762D</td>
      <td>hyper startup inc.</td>
      <td>BB099931J</td>
      <td>hyper-startup inc.</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AA098762D</td>
      <td>hyper startup inc.</td>
      <td>CC082744L</td>
      <td>hyper startup incorporated</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BB099931J</td>
      <td>hyper-startup inc.</td>
      <td>AA098762D</td>
      <td>hyper startup inc.</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BB099931J</td>
      <td>hyper-startup inc.</td>
      <td>CC082744L</td>
      <td>hyper startup incorporated</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CC082744L</td>
      <td>hyper startup incorporated</td>
      <td>BB099931J</td>
      <td>hyper-startup inc.</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CC082744L</td>
      <td>hyper startup incorporated</td>
      <td>AA098762D</td>
      <td>hyper startup inc.</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
</div>

***N.B.** the pandas index number 14 has gone because the left and right side IDs were identical.*

### Reduce data to unique rows having duplicate IDs

Finally we reduce the data to a pandas Series ready for exporting with one row for each record that has any duplicates.

```python
company_dupes = pd.DataFrame(dupes.left_side_id.unique()).squeeze().rename('company_id')
company_dupes
```

This gives the following result:

```
0    AA012345X
1    BB016741P
2    AA098762D
3    BB099931J
4    CC082744L
5    AA903844B
6    BB904941H
7    AA777431C
8    BB760431Y
9    BB750431M
Name: company_id, dtype: object
```

How this is processed, as with any database clean-up, is out of the scope of this tutorial. A first step however could be:

1. Import the list of database IDs into the relevant database as a temporary table
1. Do an inner-join with the original table the data was exported from and sort ascending by the 'name' column

This will return filtered rows with the 'name' field in adjacent rows showing similar matched strings.
