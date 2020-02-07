scopyon
=======

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ecell/scopyon/master)
[![PyPI version](https://badge.fury.io/py/scopyon.svg)](https://badge.fury.io/py/scopyon)
[![CircleCI](https://circleci.com/gh/ecell/scopyon.svg?style=svg)](https://circleci.com/gh/ecell/scopyon)
[![Readthedocs](https://readthedocs.org/projects/scopyon/badge/)](http://scopyon.readthedocs.io/)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/ca714025c04b456dbaa036e0275cb603)](https://www.codacy.com/app/ecell/scopyon?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ecell/scopyon&amp;utm_campaign=Badge_Grade)

:microscope: Monte Carlo simulation toolkit for bioimaging systems

Requirements
------------

For `scopyon`, Python 3 and its libraries, `numpy`, `scipy` and `matplotlib`, are required. `scikit-image` is also needed by `scopyon.spot_detection`.

Installation
------------

```shell-session
$ pip install scopyon
```

or

```shell-session
$ python setup.py test install
```

Quick start
-----------

```shell-session
$ python examples/twocolor_script.py
```

![TIRF Image](https://github.com/ecell/scopyon/raw/master/examples/data/outputs_tirf/twocolor_0000000.png)

License
-------

`scopyon` is licensed under the terms of BSD-3-Clause (see [LICENSE](/LICENSE)).

Citation
--------

If this package contributes to your work, please cite the followings.

[https://doi.org/10.1371/journal.pone.0130089](https://doi.org/10.1371/journal.pone.0130089)
