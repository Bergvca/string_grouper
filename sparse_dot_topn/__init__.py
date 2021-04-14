# flake8: noqa
import sys

if sys.version_info[0] >= 3:
    from sparse_dot_topn.awesome_cossim_minmax_topn import awesome_cossim_minmax_topn
else:
    from awesome_cossim_minmax_topn import awesome_cossim_minmax_topn