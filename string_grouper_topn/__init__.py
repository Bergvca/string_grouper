# flake8: noqa
import sys

if sys.version_info[0] >= 3:
    from string_grouper_topn.awesome_cossim_topn import awesome_cossim_topn
else:
    from awesome_cossim_topn import awesome_cossim_topn