from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='string_grouper',
    version='0.1.0',
    packages=['string_grouper'],
    license='MIT License',
    description='String grouper contains functions to do string matching using TF-IDF and the cossine similarity. '
                'Based on https://bergvca.github.io/2017/10/14/super-fast-string-matching.html',
    author='Chris van den Berg',
    long_description=README,
    long_description_content_type="text/markdown",
    author_email='fake_email@gmail.com',
    url='https://github.com/Bergvca/string_grouper',
    zip_safe=False,
    python_requires='>3.7',
    install_requires=['pandas>=0.25.3'
                      , 'scipy'
                      , 'sklearn'
                      , 'sparse_dot_topn>=0.2.6'
                      ]
)
