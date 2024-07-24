#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Anarya Ray
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
A bounded 3-D KDE class for all of your bounded 3-D KDE needs.
"""

import numpy as np
from scipy.stats import gaussian_kde as kde

class Bounded_3d_kde(kde):
    r"""Represents a two-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain."""

    def __init__(self, pts, low=[None,None,None],high=[None,None,None],
                 bw=None, *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param low: array of lower boundaries.

        :param high: array of upper boundaries.

        """
        pts = np.atleast_2d(pts)

        assert pts.ndim == 2, 'Bounded_kde points array can only be two-dimensional'

        super(Bounded_3d_kde, self).__init__(pts.T, bw_method=bw, *args, **kwargs)

        self.low=low
        self.high=high
    
    #     @property
    #     def low(self):
    #         """The lower bounds"""
    #         return self.low

    #     @property
    #     def xhigh(self):
    #         """The upper bound of the x domain."""
    #         return self.high

    
    

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given
        points."""
        pts = np.atleast_2d(pts)
        assert pts.ndim == 2, 'points must be two-dimensional'
        
        pts_orig=np.copy(pts)
        
        pdf = super(Bounded_3d_kde, self).evaluate(pts.T)
        for i,(low,high) in enumerate(zip(self.low,self.high)):

            
            if not np.isneginf(low) and low is not None:
                    pts[:,i] = 2.0 * low - pts[:,i]
                    pdf += super(Bounded_3d_kde, self).evaluate(pts.T)
                    pts[:,i] = pts_orig[:,i]

            if not np.isposinf(high) and  high is not None:
                    pts[:,i] = 2.0 * high - pts[:,i]
                    pdf += super(Bounded_3d_kde, self).evaluate(pts.T)
                    pts[:,i] = pts_orig[:,i]

        return pdf

    def __call__(self, pts):
        
        results = self.evaluate(pts)
        pts = np.atleast_2d(pts)
        
        out_of_bounds = np.zeros(pts.shape[0], dtype='bool')
        
        for i,(low,high) in enumerate(zip(self.low,self.high)):
            
            if not np.isneginf(low) and low is not None:
                out_of_bounds[pts[:,i]<low] =True

            if not np.isposinf(high) and  high is not None:
                out_of_bounds[pts[:,i]>high] =True
        
        
        results[out_of_bounds]=0.

        
        
        
        return results
    
    def resample(self,size=1):
        samples=super(Bounded_3d_kde, self).resample(size)
        for i,(low,high) in enumerate(zip(self.low,self.high)):
            
            if not np.isneginf(low) and low is not None:
                    
                samples[i,:][samples[i,:]<low] = 2.*low- samples[i,:][samples[i,:]<low]
                    

            if not np.isposinf(high) and  high is not None:
                    
                samples[i,:][samples[i,:]>high] = 2.*high- samples[i,:][samples[i,:]>high]

        return samples

