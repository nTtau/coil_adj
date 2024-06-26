'''
Created on Apr 29, 2013

@author: sieck
'''

from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates
from numpy import product,vstack,gradient,ones_like,zeros_like,mean
from numpy.testing import assert_allclose,assert_equal

class _B_grid():
   def __init__(self,data,r0,dr):
      self.data=data
      self.r0=r0
      self.dr=dr
   def __call__(self,r):
      r_normalized=((r-self.r0)/self.dr).transpose()
      return vstack((map_coordinates(self.data[...,0],r_normalized),
                     map_coordinates(self.data[...,1],r_normalized),
                     map_coordinates(self.data[...,2],r_normalized))).transpose()

class B_interpolator():
   def __init__(self,r,B):
      self.r=r
      self.data=B
      try:
         assert len(r) == B.shape[1], 'Wrong dimensionality for r {} or B {}'.format(r.shape, B.shape)
         for d1,obj in enumerate(r):
            assert product(obj.shape) == B.shape[0], 'Wrong number of points in r or B'
            for d2,grad in enumerate(gradient(obj)):
               if d1==d2:
                  assert_allclose(grad,mean(grad)*ones_like(grad), err_msg='Irregular grid spacing')
               else:
                  assert_equal(grad,zeros_like(grad), err_msg='Rotated grid')
         print('Creating structured interpolator')
         dr=[]
         r0=[]
         for dim in range(len(r)):
            origin_idx=()
            newpt_idx=()
            for d2 in range(len(r)):
               origin_idx += (0,)
               if dim == d2:
                  newpt_idx += (1,)
               else:
                  newpt_idx += (0,)
            origin=r[dim].__getitem__(origin_idx)
            newpt=r[dim].__getitem__(newpt_idx)
            r0.append(origin)     
            dr.append(newpt-origin)
         self.B_gen=_B_grid(self.data.reshape(r[0].shape+(B.shape[1],)), r0,dr)
         
      except Exception as e: # r must be Nx3
         print (e)
         print('Creating unstructured interpolator')
         self.B_gen=LinearNDInterpolator(r,B)
   
   def B(self,r):
      return self.B_gen(r.reshape((-1,3)))
