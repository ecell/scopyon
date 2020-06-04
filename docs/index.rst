#######################################
Welcome to scopyon's documentation!
#######################################

:code:`scopyon` is a Monte Carlo simulation toolkit for bioimagins systems.

`https://github.com/ecell/scopyon <https://github.com/ecell/scopyon>`_

**************
Prerequisites
**************

For :code:`scopyon`, Python 3 and its libraries, :code:`numpy`, :code:`scipy`, :code:`matplotlib`, :code:`pyyaml` and :code:`pint`, are required (see also `requirements.txt <https://github.com/ecell/scopyon/blob/docs/requirements.txt`>_). Installation of :code:`scikit-image`, :code:`pillow` and :code:`plotly` is also recommended.

**************
Installation
**************

::

    pip install scopyon

or

::

    git clone https://github.com/ecell/scopyon.git
    cd scopyon
    python setup.py test install

**************
Quick start
**************

::

    python examples/twocolor.py

.. image:: https://github.com/ecell/scopyon/raw/master/examples/twocolor_000.png
    :alt: TIRF Image

*************
Examples
*************

.. toctree::
    :maxdepth: 2

    examples/index

**********
License
**********

:code:`scopyon` is licensed under the terms of BSD-3-Clause (see `LICENSE <https://github.com/ecell/scopyon/blob/master/README.md>`_).

***********
Citation
***********

If this package contributes to your work, please cite the followings

`https://doi.org/10.1371/journal.pone.0130089 <https://doi.org/10.1371/journal.pone.0130089>`_

*******************
Indices and tables
*******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
