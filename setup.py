from setuptools import setup

setup(
    name='string_grouper',
    version='0.1.0',
    packages=['string_grouper'],
    license='MIT License',
    description='String grouper contains functions to do string matching using TF-IDF and the cossine similarity. '
                'Based on https://bergvca.github.io/2017/10/14/super-fast-string-matching.html',
    author='Chris van den Berg',
    author_email='fake_email@gmail.com',
    zip_safe=False,
    python_requires='>3.7',
    install_requires=['pandas'
                      , 'numpy'
                      , 'scipy'
                      , 'sklearn'
                      , 'sparse_dot_topn'
                      ]
)
