.. PyLaplace documentation master file, created by
   sphinx-quickstart on Fri Jun  1 21:12:37 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyLaplace!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

PyLaplace is a Python implementation of generalized Laplace
coefficients by three different methods. The generalized Laplace
coefficients are defined by

.. math::
        b_s^m(a) = \frac{2}{\pi}\int_0^\pi \frac{\cos(m\phi)d\phi}{(q^2 + p^2a^2 - 2a\cos(\phi))^s}

The result is determined by parameters :math:`a`, :math:`m`,
:math:`s`, :math:`q` and :math:`p`. These coefficients pop up
frequently in celestial mechanics and may need to be evaluated many
times, in which case brute force numerical integration is too
slow. Pylaplace includes two methods for faster evaluation, one based
on hypergeometric functions and an approximate method involving Bessel
functions.

Installation
-------------------------------------

If Python is not installed, download from `here
<https://www.python.org/downloads/>`_ and install. The latest versions
of Python come with package manager pip included. Then PyLaplace can be
installed from the command line simply by entering::

  pip install pylaplace

Usage
-------------------------------------

Within Python, first import the class::

  >>> from pylaplace import LaplaceCoefficient

Create an instance of LaplaceCoefficient::

  >>> laplace = LaplaceCoefficient()

The generalized Laplace coefficient can then be calculated simply as::

  >>> result = laplace(a, s, m, p, q)

The derivative with respect to :math:`a` can be calculated using::

  >>> result = laplace.derivative(a, s, m, p, q)

By default, the Laplace coefficients are calculated using
hypergeometric functions. Two other methods are available: 'Brute'
(brute force integration, very slow) and 'Bessel' (fast but
approximate). These can be selected at creation by entering::

  >>> laplace = LaplaceCoefficient(method='Brute')


Class reference
=========================

.. automodule:: pylaplace.laplace

.. autoclass:: LaplaceCoefficient
      :members:

      .. automethod:: __call__
