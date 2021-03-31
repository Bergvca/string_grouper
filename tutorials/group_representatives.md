# Group Representatives
------


```python
import pandas as pd
from string_grouper import group_similar_strings
```

We have already seen that <samp>string_grouper</samp> has a function <samp>group_similar_strings()</samp> that partitions a <samp>Series</samp> of strings into groups based on their degree of mutual similarity.  To represent each group, <samp>group_similar_strings()</samp> chooses one member of the group.  The default choice is the so-called ***centroid*** of the group.

The **centroid** of a group of similar strings is that string in the group which has the highest ***similarity aggregate***.

The **similarity aggregate** of a string is the sum of all the cosine similarities between it and the strings that it matches.

This choice can also be specified by setting the following keyword argument of <samp>group_similar_strings</samp>:
`group_rep='centroid'`.

<samp>group_similar_strings()</samp> has an alternative choice of group representative which is specified by setting `group_rep='first'`.  This choice is merely the first member of the group according to its index (that is, its position in the order of appearance of members in the group).  Though somewhat arbitrary, this choice is the fastest and can be used for large datasets whenever the choice of group-representative is not important. 

|`group_rep='first'`|
|:---:|
|**`group_rep='centroid'`**|

But the user may not be satisfied with <samp>group_similar_strings()</samp>' only two available choices. For example, he/she might prefer the earliest recorded string in the group to represent the group (if timestamp metadata is available).  Fortunately, there are three other choices available in an auxiliary module named `string_grouper_utils` included in the package and which can be imported whenever necessary:


```python
from string_grouper_utils import new_group_rep_by_highest_weight, \
    new_group_rep_by_earliest_timestamp, new_group_rep_by_completeness
```

<samp>string_grouper_utils</samp> provides three high-level functions `new_group_rep_by_highest_weight()`, `new_group_rep_by_earliest_timestamp()`, and `new_group_rep_by_completeness()`.  These functions change the group-representatives of data that have already been grouped (by <samp>group_similar_strings()</samp>, for example).

Let us create a DataFrame with some artificial timestamped records:


```python
customers_df = pd.DataFrame(
   [
      ('BB016741P', 'Mega Enterprises Corporation', 'Address0', 'Tel0', 'Description0', 0.2, '2014-12-30 10:55:00-02:00'),
      ('CC082744L', 'Hyper Startup Incorporated', '', 'Tel1', '', 0.5, '2017-01-01 20:23:15-05:00'),
      ('AA098762D', 'Hyper Startup Inc.', 'Address2', 'Tel2', 'Description2', 0.3, '2020-10-20 15:29:30+02:00'),
      ('BB099931J', 'Hyper-Startup Inc.', 'Address3', 'Tel3', 'Description3', 0.1, '2013-07-01 03:34:45-05:00'),
      ('HH072982K', 'Hyper Hyper Inc.', 'Address4', '', 'Description4', 0.9, '2005-09-11 11:56:00-07:00'),
      ('EE059082Q', 'Mega Enterprises Corp.', 'Address5', 'Tel5', 'Description5', 1.0, '1998-04-14 09:21:11+00:00')
   ],
   columns=('Customer ID', 'Customer Name', 'Address', 'Tel', 'Description', 'weight', 'timestamp')
).set_index('Customer ID')
```

**NB.** These 'timestamps' are not actual `pandas Timestamp` datatypes --- they are strings.  If we like, we could convert these strings to `pandas Timestamp` datatypes or datetime datatypes (from python module `datetime`), but this is not necessary because <samp>string_grouper_utils</samp> can deal with these strings just as they are and can automatically _parse_ them to into (localized) `pandas Timestamp` datatypes internally for comparison as we shall soon see.  

Also, in this example we have used the most general timestamps, that is, each string has a date together with time-of-day and timezone information.  This is not always necessary, for example, if desired, only date information can be contained in each string.

Let us display the DataFrame:


```python
customers_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer Name</th>
      <th>Address</th>
      <th>Tel</th>
      <th>Description</th>
      <th>weight</th>
      <th>timestamp</th>
    </tr>
    <tr>
      <th>Customer ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BB016741P</th>
      <td>Mega Enterprises Corporation</td>
      <td>Address0</td>
      <td>Tel0</td>
      <td>Description0</td>
      <td>0.2</td>
      <td>2014-12-30 10:55:00-02:00</td>
    </tr>
    <tr>
      <th>CC082744L</th>
      <td>Hyper Startup Incorporated</td>
      <td></td>
      <td>Tel1</td>
      <td></td>
      <td>0.5</td>
      <td>2017-01-01 20:23:15-05:00</td>
    </tr>
    <tr>
      <th>AA098762D</th>
      <td>Hyper Startup Inc.</td>
      <td>Address2</td>
      <td>Tel2</td>
      <td>Description2</td>
      <td>0.3</td>
      <td>2020-10-20 15:29:30+02:00</td>
    </tr>
    <tr>
      <th>BB099931J</th>
      <td>Hyper-Startup Inc.</td>
      <td>Address3</td>
      <td>Tel3</td>
      <td>Description3</td>
      <td>0.1</td>
      <td>2013-07-01 03:34:45-05:00</td>
    </tr>
    <tr>
      <th>HH072982K</th>
      <td>Hyper Hyper Inc.</td>
      <td>Address4</td>
      <td></td>
      <td>Description4</td>
      <td>0.9</td>
      <td>2005-09-11 11:56:00-07:00</td>
    </tr>
    <tr>
      <th>EE059082Q</th>
      <td>Mega Enterprises Corp.</td>
      <td>Address5</td>
      <td>Tel5</td>
      <td>Description5</td>
      <td>1.0</td>
      <td>1998-04-14 09:21:11+00:00</td>
    </tr>
  </tbody>
</table>
</div>



## <samp>group_similar_strings()</samp>

With the following command, we can create a mapping table with the groupings that <samp>group_similar_strings()</samp> finds.  Here the keyword argument `group_rep` is not explicitly set.  It therefore takes on the default value `'centroid'`. 


```python
customers_df[['group rep ID', 'group rep']] = \
    group_similar_strings(
        customers_df['Customer Name'], 
        min_similarity=0.6)
```

Let's display the mapping table:


```python
customers_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer Name</th>
      <th>Address</th>
      <th>Tel</th>
      <th>Description</th>
      <th>weight</th>
      <th>timestamp</th>
      <th>group rep ID</th>
      <th>group rep</th>
    </tr>
    <tr>
      <th>Customer ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BB016741P</th>
      <td>Mega Enterprises Corporation</td>
      <td>Address0</td>
      <td>Tel0</td>
      <td>Description0</td>
      <td>0.2</td>
      <td>2014-12-30 10:55:00-02:00</td>
      <td>BB016741P</td>
      <td>Mega Enterprises Corporation</td>
    </tr>
    <tr>
      <th>CC082744L</th>
      <td>Hyper Startup Incorporated</td>
      <td></td>
      <td>Tel1</td>
      <td></td>
      <td>0.5</td>
      <td>2017-01-01 20:23:15-05:00</td>
      <td>AA098762D</td>
      <td>Hyper Startup Inc.</td>
    </tr>
    <tr>
      <th>AA098762D</th>
      <td>Hyper Startup Inc.</td>
      <td>Address2</td>
      <td>Tel2</td>
      <td>Description2</td>
      <td>0.3</td>
      <td>2020-10-20 15:29:30+02:00</td>
      <td>AA098762D</td>
      <td>Hyper Startup Inc.</td>
    </tr>
    <tr>
      <th>BB099931J</th>
      <td>Hyper-Startup Inc.</td>
      <td>Address3</td>
      <td>Tel3</td>
      <td>Description3</td>
      <td>0.1</td>
      <td>2013-07-01 03:34:45-05:00</td>
      <td>AA098762D</td>
      <td>Hyper Startup Inc.</td>
    </tr>
    <tr>
      <th>HH072982K</th>
      <td>Hyper Hyper Inc.</td>
      <td>Address4</td>
      <td></td>
      <td>Description4</td>
      <td>0.9</td>
      <td>2005-09-11 11:56:00-07:00</td>
      <td>HH072982K</td>
      <td>Hyper Hyper Inc.</td>
    </tr>
    <tr>
      <th>EE059082Q</th>
      <td>Mega Enterprises Corp.</td>
      <td>Address5</td>
      <td>Tel5</td>
      <td>Description5</td>
      <td>1.0</td>
      <td>1998-04-14 09:21:11+00:00</td>
      <td>BB016741P</td>
      <td>Mega Enterprises Corporation</td>
    </tr>
  </tbody>
</table>
</div>



Let's try this again, this time with <samp>group_rep='first'</samp>:


```python
customers_df[['group rep ID', 'group rep']] = \
    group_similar_strings(
        customers_df['Customer Name'], 
        group_rep='first', 
        min_similarity=0.6)
```

Displaying the new mapping table shows the differences from the result above:


```python
customers_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer Name</th>
      <th>Address</th>
      <th>Tel</th>
      <th>Description</th>
      <th>weight</th>
      <th>timestamp</th>
      <th>group rep ID</th>
      <th>group rep</th>
    </tr>
    <tr>
      <th>Customer ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BB016741P</th>
      <td>Mega Enterprises Corporation</td>
      <td>Address0</td>
      <td>Tel0</td>
      <td>Description0</td>
      <td>0.2</td>
      <td>2014-12-30 10:55:00-02:00</td>
      <td>BB016741P</td>
      <td>Mega Enterprises Corporation</td>
    </tr>
    <tr>
      <th>CC082744L</th>
      <td>Hyper Startup Incorporated</td>
      <td></td>
      <td>Tel1</td>
      <td></td>
      <td>0.5</td>
      <td>2017-01-01 20:23:15-05:00</td>
      <td>CC082744L</td>
      <td>Hyper Startup Incorporated</td>
    </tr>
    <tr>
      <th>AA098762D</th>
      <td>Hyper Startup Inc.</td>
      <td>Address2</td>
      <td>Tel2</td>
      <td>Description2</td>
      <td>0.3</td>
      <td>2020-10-20 15:29:30+02:00</td>
      <td>CC082744L</td>
      <td>Hyper Startup Incorporated</td>
    </tr>
    <tr>
      <th>BB099931J</th>
      <td>Hyper-Startup Inc.</td>
      <td>Address3</td>
      <td>Tel3</td>
      <td>Description3</td>
      <td>0.1</td>
      <td>2013-07-01 03:34:45-05:00</td>
      <td>CC082744L</td>
      <td>Hyper Startup Incorporated</td>
    </tr>
    <tr>
      <th>HH072982K</th>
      <td>Hyper Hyper Inc.</td>
      <td>Address4</td>
      <td></td>
      <td>Description4</td>
      <td>0.9</td>
      <td>2005-09-11 11:56:00-07:00</td>
      <td>HH072982K</td>
      <td>Hyper Hyper Inc.</td>
    </tr>
    <tr>
      <th>EE059082Q</th>
      <td>Mega Enterprises Corp.</td>
      <td>Address5</td>
      <td>Tel5</td>
      <td>Description5</td>
      <td>1.0</td>
      <td>1998-04-14 09:21:11+00:00</td>
      <td>BB016741P</td>
      <td>Mega Enterprises Corporation</td>
    </tr>
  </tbody>
</table>
</div>



Remember it displays the same groups!  Only the group names (representatives) have changed.

## <samp>new_group_rep_by_earliest_timestamp()</samp>

As mentioned above, there are still more choices of group-representatives available.  Let's use the `new_group_rep_by_earliest_timestamp()` function:


```python
customers_df.reset_index(inplace=True)
customers_df[['group rep ID', 'group rep']] = \
    new_group_rep_by_earliest_timestamp(
        grouped_data=customers_df,
        group_col='group rep ID',
        record_id_col='Customer ID', 
        record_name_col='Customer Name',
        timestamps='timestamp',
        dayfirst=False
)
```

Notice that this time ***the function operates on already grouped data*** (such as the mapping table that was output by <samp>group_similar_strings()</samp> above).  Thus ***the column of the input grouped data containing the groups*** (here either 'group rep ID' or 'group rep') ***must be specified as argument <samp>group_col</samp> in addition to the column containing the group members*** (here either 'Customer ID' or 'Customer Name') ***in argument <samp>record_id_col</samp>***.  

Argument <samp>record_name_col</samp> is optional and will appear in the output alongside the new group-representatives chosen from <samp>record_id_col</samp> only if specified.

The keyword argument `dayfirst` used here is one that is also used in python module <samp>dateutil</samp>'s <samp>parser.parse()</samp> function. This option specifies whether to interpret the first value in an ambiguous 3-integer date (e.g. 01/05/09) as the day ('True') or month ('False'). If keyword argument `yearfirst` is set to 'True', this distinguishes between YDM and YMD. 

The other possible keyword arguments that can be used are detailed in the docstring (help) of <samp>new_group_rep_by_earliest_timestamp()</samp>:


```python
help(new_group_rep_by_earliest_timestamp)
```

    Help on function new_group_rep_by_earliest_timestamp in module string_grouper_utils.string_grouper_utils:
    
    new_group_rep_by_earliest_timestamp(grouped_data: pandas.core.frame.DataFrame, group_col: Union[str, int], record_id_col: Union[str, int], timestamps: Union[pandas.core.series.Series, str, int], record_name_col: Union[str, int, NoneType] = None, parserinfo=None, **kwargs) -> Union[pandas.core.frame.DataFrame, pandas.core.series.Series]
        Selects the oldest string in each group as group-representative.
        :param grouped_data: The grouped DataFrame
        :param group_col: The name or positional index of the column in grouped_data containing the groups
        :param record_id_col: The name or positional index of the column in grouped_data with all groups' members' IDs
        (This will appear in the output)
        :param timestamps: pandas.Series or the column name (str) or column positional index (int) in grouped_data
        This contains the timestamps of the strings to be grouped.
        :param record_name_col: (Optional) The name or positional index of the column in grouped_data with
        all groups' members' names. (This will appear in the output.)
        :param parserinfo: (See below.)
        :param **kwargs: (See below.)
        parserinfo and kwargs are the same arguments as those you would pass to dateutil.parser.parse.  They help in
        interpreting the string inputs which are to be parsed into datetime datatypes.
        
        FYI, the dateutil.parser.parse documentation for these arguments follows:
        :param parserinfo:
            A :class:`parserinfo` object containing parameters for the parser.
            If ``None``, the default arguments to the :class:`parserinfo`
            constructor are used.
        
        The ``**kwargs`` parameter takes the following keyword arguments:
        
        :param default:
            The default datetime object, if this is a datetime object and not
            ``None``, elements specified in the strings containing the date/time-stamps replace elements in the
            default object.
        
        :param ignoretz:
            If set ``True``, time zones in parsed strings are ignored and a naive
            :class:`datetime` object is returned.
        
        :param tzinfos:
            Additional time zone names / aliases which may be present in the
            string. This argument maps time zone names (and optionally offsets
            from those time zones) to time zones. This parameter can be a
            dictionary with timezone aliases mapping time zone names to time
            zones or a function taking two parameters (``tzname`` and
            ``tzoffset``) and returning a time zone.
        
            The timezones to which the names are mapped can be an integer
            offset from UTC in seconds or a :class:`tzinfo` object.
        
            .. doctest::
               :options: +NORMALIZE_WHITESPACE
        
                >>> from dateutil.parser import parse
                >>> from dateutil.tz import gettz
                >>> tzinfos = {"BRST": -7200, "CST": gettz("America/Chicago")}
                >>> parse("2012-01-19 17:21:00 BRST", tzinfos=tzinfos)
                datetime.datetime(2012, 1, 19, 17, 21, tzinfo=tzoffset(u'BRST', -7200))
                >>> parse("2012-01-19 17:21:00 CST", tzinfos=tzinfos)
                datetime.datetime(2012, 1, 19, 17, 21,
                                  tzinfo=tzfile('/usr/share/zoneinfo/America/Chicago'))
        
            This parameter is ignored if ``ignoretz`` is set.
        
        :param dayfirst:
            Whether to interpret the first value in an ambiguous 3-integer date
            (e.g. 01/05/09) as the day (``True``) or month (``False``). If
            ``yearfirst`` is set to ``True``, this distinguishes between YDM and
            YMD. If set to ``None``, this value is retrieved from the current
            :class:`parserinfo` object (which itself defaults to ``False``).
        
        :param yearfirst:
            Whether to interpret the first value in an ambiguous 3-integer date
            (e.g. 01/05/09) as the year. If ``True``, the first number is taken to
            be the year, otherwise the last number is taken to be the year. If
            this is set to ``None``, the value is retrieved from the current
            :class:`parserinfo` object (which itself defaults to ``False``).
        
        :param fuzzy:
            Whether to allow fuzzy parsing, allowing for string like "Today is
            January 1, 2047 at 8:21:00AM".
        
        :param fuzzy_with_tokens:
            If ``True``, ``fuzzy`` is automatically set to True, and the parser
            will return a tuple where the first element is the parsed
            :class:`datetime.datetime` datetimestamp and the second element is
            a tuple containing the portions of the string which were ignored:
        
            .. doctest::
        
                >>> from dateutil.parser import parse
                >>> parse("Today is January 1, 2047 at 8:21:00AM", fuzzy_with_tokens=True)
                (datetime.datetime(2047, 1, 1, 8, 21), (u'Today is ', u' ', u'at '))
    
    


```python
customers_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>Address</th>
      <th>Tel</th>
      <th>Description</th>
      <th>weight</th>
      <th>timestamp</th>
      <th>group rep ID</th>
      <th>group rep</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BB016741P</td>
      <td>Mega Enterprises Corporation</td>
      <td>Address0</td>
      <td>Tel0</td>
      <td>Description0</td>
      <td>0.2</td>
      <td>2014-12-30 10:55:00-02:00</td>
      <td>EE059082Q</td>
      <td>Mega Enterprises Corp.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CC082744L</td>
      <td>Hyper Startup Incorporated</td>
      <td></td>
      <td>Tel1</td>
      <td></td>
      <td>0.5</td>
      <td>2017-01-01 20:23:15-05:00</td>
      <td>BB099931J</td>
      <td>Hyper-Startup Inc.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AA098762D</td>
      <td>Hyper Startup Inc.</td>
      <td>Address2</td>
      <td>Tel2</td>
      <td>Description2</td>
      <td>0.3</td>
      <td>2020-10-20 15:29:30+02:00</td>
      <td>BB099931J</td>
      <td>Hyper-Startup Inc.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BB099931J</td>
      <td>Hyper-Startup Inc.</td>
      <td>Address3</td>
      <td>Tel3</td>
      <td>Description3</td>
      <td>0.1</td>
      <td>2013-07-01 03:34:45-05:00</td>
      <td>BB099931J</td>
      <td>Hyper-Startup Inc.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HH072982K</td>
      <td>Hyper Hyper Inc.</td>
      <td>Address4</td>
      <td></td>
      <td>Description4</td>
      <td>0.9</td>
      <td>2005-09-11 11:56:00-07:00</td>
      <td>HH072982K</td>
      <td>Hyper Hyper Inc.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>EE059082Q</td>
      <td>Mega Enterprises Corp.</td>
      <td>Address5</td>
      <td>Tel5</td>
      <td>Description5</td>
      <td>1.0</td>
      <td>1998-04-14 09:21:11+00:00</td>
      <td>EE059082Q</td>
      <td>Mega Enterprises Corp.</td>
    </tr>
  </tbody>
</table>
</div>



Here the group-member with the earliest timestamp has been chosen as group-representative for each group.  Notice that even though the timestamp data is input as strings, the function is able to treat them as if they were <samp>datetime</samp> (or <samp>pandas Timestamp<samp>) datatypes.

## <samp>new_group_rep_by_highest_weight()</samp> and <samp>new_group_rep_by_completeness()</samp>

The other two utility functions `new_group_rep_by_highest_weight()` and `new_group_rep_by_completeness()` operate in a similar way to <samp>new_group_rep_by_earliest_timestamp()</samp>:

1. <samp>new_group_rep_by_highest_weight()</samp> chooses the group-member with the highest weight as group-representative for each group.  The weight of each member is assigned as desired by the user, and provided as an argument to the function.  The weights could also be a specified column in the input grouped data (mapping table).

2. <samp>new_group_rep_by_completeness()</samp> chooses the group member with the most filled-in fields in its row as group-representative for each group.


```python
customers_df[['group rep ID', 'group rep']] = \
    new_group_rep_by_highest_weight(
        grouped_data=customers_df,
        group_col='group rep ID',
        record_id_col='Customer ID', 
        weights='weight',
        record_name_col='Customer Name'
)
```


```python
customers_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>Address</th>
      <th>Tel</th>
      <th>Description</th>
      <th>weight</th>
      <th>timestamp</th>
      <th>group rep ID</th>
      <th>group rep</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BB016741P</td>
      <td>Mega Enterprises Corporation</td>
      <td>Address0</td>
      <td>Tel0</td>
      <td>Description0</td>
      <td>0.2</td>
      <td>2014-12-30 10:55:00-02:00</td>
      <td>EE059082Q</td>
      <td>Mega Enterprises Corp.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CC082744L</td>
      <td>Hyper Startup Incorporated</td>
      <td></td>
      <td>Tel1</td>
      <td></td>
      <td>0.5</td>
      <td>2017-01-01 20:23:15-05:00</td>
      <td>CC082744L</td>
      <td>Hyper Startup Incorporated</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AA098762D</td>
      <td>Hyper Startup Inc.</td>
      <td>Address2</td>
      <td>Tel2</td>
      <td>Description2</td>
      <td>0.3</td>
      <td>2020-10-20 15:29:30+02:00</td>
      <td>CC082744L</td>
      <td>Hyper Startup Incorporated</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BB099931J</td>
      <td>Hyper-Startup Inc.</td>
      <td>Address3</td>
      <td>Tel3</td>
      <td>Description3</td>
      <td>0.1</td>
      <td>2013-07-01 03:34:45-05:00</td>
      <td>CC082744L</td>
      <td>Hyper Startup Incorporated</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HH072982K</td>
      <td>Hyper Hyper Inc.</td>
      <td>Address4</td>
      <td></td>
      <td>Description4</td>
      <td>0.9</td>
      <td>2005-09-11 11:56:00-07:00</td>
      <td>HH072982K</td>
      <td>Hyper Hyper Inc.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>EE059082Q</td>
      <td>Mega Enterprises Corp.</td>
      <td>Address5</td>
      <td>Tel5</td>
      <td>Description5</td>
      <td>1.0</td>
      <td>1998-04-14 09:21:11+00:00</td>
      <td>EE059082Q</td>
      <td>Mega Enterprises Corp.</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers_df[['group rep ID', 'group rep']] = \
    new_group_rep_by_completeness(
        grouped_data=customers_df,
        group_col='group rep ID',
        record_id_col='Customer ID', 
        record_name_col='Customer Name',
        tested_cols=['Address', 'Tel', 'Description']
)
```

**N.B.** If argument <samp>tesed_cols</samp> is not given, <samp>new_group_rep_by_completeness()</samp> will test the filled-in status of all the fields of <samp>grouped_data</samp> for each group member.


```python
customers_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>Address</th>
      <th>Tel</th>
      <th>Description</th>
      <th>weight</th>
      <th>timestamp</th>
      <th>group rep ID</th>
      <th>group rep</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BB016741P</td>
      <td>Mega Enterprises Corporation</td>
      <td>Address0</td>
      <td>Tel0</td>
      <td>Description0</td>
      <td>0.2</td>
      <td>2014-12-30 10:55:00-02:00</td>
      <td>BB016741P</td>
      <td>Mega Enterprises Corporation</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CC082744L</td>
      <td>Hyper Startup Incorporated</td>
      <td></td>
      <td>Tel1</td>
      <td></td>
      <td>0.5</td>
      <td>2017-01-01 20:23:15-05:00</td>
      <td>AA098762D</td>
      <td>Hyper Startup Inc.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AA098762D</td>
      <td>Hyper Startup Inc.</td>
      <td>Address2</td>
      <td>Tel2</td>
      <td>Description2</td>
      <td>0.3</td>
      <td>2020-10-20 15:29:30+02:00</td>
      <td>AA098762D</td>
      <td>Hyper Startup Inc.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BB099931J</td>
      <td>Hyper-Startup Inc.</td>
      <td>Address3</td>
      <td>Tel3</td>
      <td>Description3</td>
      <td>0.1</td>
      <td>2013-07-01 03:34:45-05:00</td>
      <td>AA098762D</td>
      <td>Hyper Startup Inc.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HH072982K</td>
      <td>Hyper Hyper Inc.</td>
      <td>Address4</td>
      <td></td>
      <td>Description4</td>
      <td>0.9</td>
      <td>2005-09-11 11:56:00-07:00</td>
      <td>HH072982K</td>
      <td>Hyper Hyper Inc.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>EE059082Q</td>
      <td>Mega Enterprises Corp.</td>
      <td>Address5</td>
      <td>Tel5</td>
      <td>Description5</td>
      <td>1.0</td>
      <td>1998-04-14 09:21:11+00:00</td>
      <td>BB016741P</td>
      <td>Mega Enterprises Corporation</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
