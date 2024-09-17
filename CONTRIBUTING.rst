=====================
Contribution Guide
=====================

Contributions are highly welcomed and appreciated. Every little help counts,
so do not hesitate! You can make a high impact on ``pysonde`` just by using
it and reporting `issues <https://github.com/observingClouds/pysonde/issues>`__.

The following sections cover some general guidelines
regarding development in ``pysonde`` for maintainers and contributors.


Nothing here is set in stone and can't be changed.
Feel free to suggest improvements or changes in the workflow.


.. _submitfeedback:

Feature requests and feedback
-----------------------------

We are eager to hear about your requests for new features and any suggestions
about the API, infrastructure, and so on. Feel free to submit these as
`issues <https://github.com/observingClouds/pysonde/issues/new>`__ with the label
``"enhancement"``.

Please make sure to explain in detail how the feature should work and keep the
scope as narrow as possible. This will make it easier to implement in small
PRs.


.. _reportbugs:

Report bugs
-----------

Report bugs for ``pysonde`` in the
`issue tracker <https://github.com/observingClouds/pysonde/issues>`_ with the
label "bug".

If you are reporting a bug, please include:

* Any details about your local setup that might be helpful in troubleshooting,
  specifically the Python interpreter version, installed libraries, and
  ``pysonde`` version.
* Detailed steps `how to reproduce the bug <https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports>__`

If you can write a demonstration test that currently fails but should pass,
that is a very useful commit to make as well, even if you cannot fix the bug
itself.


.. _fixbugs:

Bug Fix
-------

Look through the
`GitHub issues for bugs <https://github.com/observingClouds/pysonde/labels/bug>`_.

Talk to developers to find out how you can fix specific bugs.



Preparing Pull Requests
-----------------------

#. Fork the `pysonde GitHub repository <https://github.com/observingClouds/pysonde>`__.
   It's fine to use ``pysonde`` as your fork repository name because it will
   live under your user.

#. Clone your fork locally using `git <https://git-scm.com/>`_, connect your
   repository to the upstream (main project), and create a branch::

    $ git clone git@github.com:YOUR_GITHUB_USERNAME/pysonde.git
    $ cd pysonde
    $ git remote add upstream git@github.com:observingClouds/pysonde.git

    # now, to fix a bug or add feature create your own branch off "main":

    $ git checkout -b your-bugfix-feature-branch-name main

   If you need some help with Git, follow this quick start
   `guide <https://git.wiki.kernel.org/index.php/QuickStart>`_.

#. Make an editable install of ``pysonde`` by running::

    $ pip install -e .

  The PDM package manager should now also have installed the pre-commit hooks
  that ensures that we all adhere to the same coding standard.
  ``pre-commit`` automatically beautifies the code, makes it more
   maintainable and catches syntax errors.

  In case it has not been installed automatically, the next step ensures it.

#. Install `pre-commit <https://pre-commit.com>`_ and its hook on the
   ``pysonde`` repo::

     $ pip install pre-commit
     $ pre-commit install

   Afterwards ``pre-commit`` will run whenever you commit and can also be
   run independently by ``pre-commit run --all``.

   You can now edit your local working copy and run/add tests as necessary.
   Please try to follow
   `PEP-8 <https://www.python.org/dev/peps/pep-0008/#naming-conventions>`_ for
   naming. When committing, ``pre-commit`` will modify the files as
   needed, or will generally be quite clear about what you need to do to pass
   the commit test.

   ``pre-commit`` also runs::

    * `ruff <https://docs.astral.sh/ruff/>`_ code formatter.
    * `black <https://black.readthedocs.io/en/stable/>`_ code formatting
  ..


#. Break your edits up into reasonably sized commits::

    $ git commit -m "<commit message>"
    $ git push -u

#. Run all tests

   Once commits are pushed to ``origin``, GitHub Actions runs continuous
   integration of all tests with `pytest <https://docs.pytest.org/en/7.1.x/getting-started.html#get-started>`__ on all new commits.
   However, you can already run tests locally::

    $ pytest  # all
    $ pytest tests/test_mwx_output.py::test_mwx_conversion_to_level1  # specific tests

   Please stick to
   `xarray <http://xarray.pydata.org/en/stable/contributing.html>`_'s testing
   recommendations.

#. Create a new changelog entry in `CHANGELOG.rst <CHANGELOG.rst>`_:

   The entry should be entered as:

   ``<description>`` (``:pr:`#<pull request number>```) ```<author's names>`_``

   where ``<description>`` is the description of the PR related to the change
   and ``<pull request number>`` is the pull request number and
   ``<author's names>`` are your first and last names.

   Add yourself to list of authors at the end of `.zenodo.json <.zenodo.json>`_ file if
   not there yet, in alphabetical order.

#. Finally, submit a `Pull Request <https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_ through the GitHub website using this data::

    head-fork: YOUR_GITHUB_USERNAME/pysonde
    compare: your-branch-name

    base-fork: observingClouds/pysonde
    base: main

Note that you can create the ``Pull Request`` while you're working on this.
The PR will update as you add more commits. ``pysonde`` developers and
contributors can then review your code and offer suggestions.
