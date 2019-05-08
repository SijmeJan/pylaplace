# -*- coding: utf-8 -*-
"""Class dealing with calculating generalized Laplace coefficients

Generalized Laplace coefficients can be calculated either by brute force integration (slow), hypergeometric functions (faster), or Bessel fuctions (fastest but approximate).

"""

from __future__ import print_function

import numpy as np
import scipy.integrate as integrate
import scipy.special as sp

class LaplaceCoefficient():
    """Class containing functions to calculate generalized Laplace coefficients.

    The generalized Laplace coefficients are defined by

    .. math::
        b_s^m(a) = \\frac{2}{\pi}\int_0^\pi \\frac{\cos(m\phi)d\phi}{(q^2 + p^2a^2 - 2a\cos(\phi))^s}


    Args:
        method (str): way to calculate the coefficients, either 'Brute' (brute force integration, slow but exact), 'Hyper' (hypergeometric functions, faster and exact), or 'Bessel' (Bessel functions, fastest but approximate)

    """

    def __init__(self, method='Hyper'):
        if (method != 'Bessel' and
            method != 'Hyper' and
            method != 'Brute'):
            print("Error: method should be either 'Bessel', 'Hyper', or 'Brute'")

        # Set functions according to method used
        self._calc = None
        self._calc_derivative = None
        if method == 'Bessel':
            self._calc = self._laplace_bessel
            self._calc_derivative = self._laplace_derivative_bessel
        if method == 'Hyper':
            self._calc = self._laplace_hyper
            self._calc_derivative = self._laplace_derivative_hyper
        if method == 'Brute':
            self._calc = self._laplace_brute
            self._calc_derivative = self._laplace_derivative_brute

    def __call__(self, a, s, m, p, q):
        """Calculate generalized Laplace coefficient.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        return self._calc(a, s, m, p, q)

    def derivative(self, a, s, m, p, q):
        """Calculate derivative with respect to :math:`a` of generalized Laplace coefficient.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        return self._calc_derivative(a, s, m, p, q)

    def _laplace_brute(self, a, s, m, p, q):
        """Calculate generalized Laplace coefficient by brute force integration.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """

        f = lambda x: np.cos(m*x)*np.power(p*p*a*a - 2.0*a*np.cos(x) + q*q, -s)
        res = integrate.quad(f, 0, np.pi, limit=100)
        ret = 2.0*res[0]/np.pi

        return ret

    def _laplace_derivative_brute(self, a, s, m, p, q):
        """Calculate derivative of generalized Laplace coefficient by brute force integration.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        f = lambda x: -2.0*s*np.cos(m*x)* \
          np.power(p*p*a*a - 2.0*a*np.cos(x) + q*q, -s-1.0)*(p*p*a - np.cos(x))
        res = integrate.quad(f, 0, np.pi, limit=100)
        ret = 2.0*res[0]/np.pi

        return ret

    def _laplace_bessel(self, a, s, m, p, q):
        """Calculate generalized Laplace coefficient using Bessel functions.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        K = sp.kn(0, m*np.sqrt((q*q + p*p*a*a - 2.0*a)/a))

        return 2.0*K/(np.pi*np.sqrt(a))

    def _laplace_derivative_bessel(self, a, s, m, p, q):
        """Calculate derivative of generalized Laplace coefficient using Bessel functions.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        K = sp.kn(1, m*np.sqrt((q*q + p*p*a*a - 2.0*a)/a))
        dxda = 0.5*m*(p*p*a*a  - q*q)/(a*a)/np.sqrt((q*q + p*p*a*a - 2.0*a)/a)
        ret = -0.5*self._laplace_bessel(a, s, m, p, q)/a - \
          2.0*K*dxda/(np.pi*np.sqrt(a))

        return ret

    def _laplace_hyper(self, a, s, m, p, q):
        """Calculate generalized Laplace coefficient using hypergeometric functions.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        ia = 1.0/(a + 1.0e-30)
        b = self._beta(a, p ,q)
        F = sp.hyp2f1(s, m + s, m + 1, b*b)

        if np.isnan(F):
            return self._laplace_brute(a, s, m, p, q)

        return 2.0*sp.binom(-s, m)*np.power(b, m)*np.power(b*ia, s)*F

    def _laplace_derivative_hyper(self, a, s, m, p, q):
        """Calculate derivative of generalized Laplace coefficient using hypergeometric functions.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            s (float): power to which denominator is raised
            m (float): numerator of integrant is :math:`\cos(m\phi)`
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        b = self._beta(a, p ,q)
        dbda = self._dbeta(a, p, q)
        ia = 1.0/(a + 1.0e-30)
        ib = 1.0/(b + 1.0e-30)

        dF = s*(m+s)*sp.hyp2f1(s + 1, m + s + 1, m + 2, b*b)/(m + 1.0)
        F = self._laplace_hyper(a, s, m, p, q)

        if (np.isnan(F) or np.isnan(dF)):
            return self._laplace_derivative_brute(a, s, m, p, q)

        return (((m+s)*ib*dbda - s*ia)*F +
                4.0*np.power(b, m+1)*np.power(b*ia, s)*sp.binom(-s, m)*dF*dbda)

    def _beta(self, a, p, q):
        """Helper function for hypergeometric method.

        Calculate beta, which is the square root of the required argument of the hypergeometric function.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        ia = 1.0/(a + 1.0e-30)
        return 0.5*ia*(q*q + p*p*a*a -
                    np.sqrt((q*q + 2.0*a + p*p*a*a)*(q*q - 2.0*a + p*p*a*a)))

    def _dbeta(self, a, p, q):
        """Helper function for hypergeometric method.

        Calculate the derivative of beta, where beta is the square root of the required argument of the hypergeometric function.

        Args:
            a (float): radius-like coordinate, must be smaller than unity
            p (float): factor multiplying :math:`a^2` in denominator
            q (float): constant term in denominator

        """
        ia = 1.0/(a + 1.0e-30)
        sq = np.sqrt((q*q + 2.0*a + p*p*a*a)*(q*q - 2.0*a + p*p*a*a))
        b = 0.5*ia*(q*q + p*p*a*a - sq)

        return p*p - b*ia - ((q*q + p*p*a*a)*p*p - 2.0)/sq
