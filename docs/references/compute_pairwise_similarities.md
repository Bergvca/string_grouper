---
title: compute_pairwise_similarities
---

Returns a `Series` of cosine similarity scores the same length and index as `string_series_1`.  Each score is the cosine similarity between the pair of strings in the same position (row) in the two input `Series`, `string_series_1` and `string_series_2`, as the position of the score in the output `Series`.  This can be seen as an element-wise comparison between the two input `Series`.
   

All functions are built using a class **`StringGrouper`**. This class can be used through pre-defined functions, for example the four high level functions above, as well as using a more interactive approach where matches can be added or removed if needed by calling the **`StringGrouper`** class directly.
   

