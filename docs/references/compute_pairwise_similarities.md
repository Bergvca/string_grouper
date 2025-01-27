---
title: compute_pairwise_similarities
---


## Arguments

```python
compute_pairwise_similarities(string_series_1: pd.Series,
                              string_series_2: pd.Series,
                              **kwargs) -> pd.Series
```


## Result

Returns a `Series` of cosine similarity scores the same length and index as `string_series_1`.  Each score is the cosine similarity between the pair of strings in the same position (row) in the two input `Series`, `string_series_1` and `string_series_2`, as the position of the score in the output `Series`.  This can be seen as an element-wise comparison between the two input `Series`.
   


