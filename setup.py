from setuptools import setup, Extension
import pathlib
import os

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# workaround for numpy and Cython install dependency
# the solution is from https://stackoverflow.com/a/54138355
def my_build_ext(pars):
    # import delayed:
    from setuptools.command.build_ext import build_ext as _build_ext
    class build_ext(_build_ext):
        def finalize_options(self):
            _build_ext.finalize_options(self)
            # Prevent numpy from thinking it is still in its setup process:
            __builtins__.__NUMPY_SETUP__ = False
            import numpy
            self.include_dirs.append(numpy.get_include())

    #object returned:
    return build_ext(pars)

if os.name == 'nt':
    extra_compile_args = ["-Ox"]
else:
    extra_compile_args = ['-std=c++0x', '-pthread', '-O3']

original_ext = Extension('sparse_dot_topn.sparse_dot_topn',
                         sources=[
                                    './sparse_dot_topn/sparse_dot_topn.pyx',
                                    './sparse_dot_topn/sparse_dot_topn_source.cpp'
                                ],
                         extra_compile_args=extra_compile_args,
                         define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                         language='c++')

threaded_ext = Extension('sparse_dot_topn.sparse_dot_topn_threaded',
                         sources=[
                             './sparse_dot_topn/sparse_dot_topn_threaded.pyx',
                             './sparse_dot_topn/sparse_dot_topn_source.cpp',
                             './sparse_dot_topn/sparse_dot_topn_parallel.cpp'
                            ],
                         extra_compile_args=extra_compile_args,
                         define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                         language='c++')

setup(
    name='string_grouper',
    version='0.4.0',
    packages=[
        'string_grouper'
        , 'string_grouper_utils'
        , 'sparse_dot_topn'
    ],
    license='MIT License',
    description='String grouper contains functions to do string matching using TF-IDF and the cossine similarity. '
                'Based on https://bergvca.github.io/2017/10/14/super-fast-string-matching.html',
    keywords='cosine-similarity sparse-matrix sparse-graph scipy cython',
    author='Chris van den Berg',
    long_description=README,
    long_description_content_type="text/markdown",
    author_email='fake_email@gmail.com',
    url='https://github.com/Bergvca/string_grouper',
    zip_safe=False,
    python_requires='>3.7',
    setup_requires=[# Setuptools 18.0 properly handles Cython extensions.
                    'setuptools>=18.0'
                    , 'cython>=0.29.15'
                    , 'numpy'
                    , 'scipy'
    ],
    install_requires=[# Setuptools 18.0 properly handles Cython extensions.
                      'setuptools>=18.0'
                      , 'cython>=0.29.15'
                      , 'numpy'
                      , 'scipy'
                      , 'scikit-learn'
                      , 'pandas>=0.25.3'
    ],
    cmdclass={'build_ext': my_build_ext},
    ext_modules=[original_ext, threaded_ext]
)
