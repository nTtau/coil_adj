'''
Created on Oct 1, 2012

@author: sieck
'''
from multiprocessing import Manager, Pool,cpu_count
import logging
import os

yes_I_have_wrapped_my_main=False
warning_done=False

verbose=False
n_procs=cpu_count()-1 # defaults to use almost all available compute power, leaving one core open to keep machine usable
logger=logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

inQ=None
outQ=None
logQ=None
MagCalcPool=None

class _MagCalcDone():
   pass

def _Bcalc(r,inq,outq,func_name,logQ):
   # TODO: some log handler that takes messages & puts them into the logQ, and some server in the main proc. to collect them
   B=0
   coil=inq.get()
   while not isinstance(coil,_MagCalcDone):
      Bfunc=getattr(coil,func_name)
      B += Bfunc(r)
      coil=inq.get()
   outq.put(B)

def inQ_put(inQ,collection):
   try:
      for thing in collection:
         inQ_put(inQ,thing)
   except: # not a collection
      inQ.put(collection)

def MultiProcMagCalc(coils,func_name,r,initial=0):
   global warning_done, inQ,outQ,logQ,MagCalcPool
   B=initial
   if (n_procs==1) or (os.name=='nt' and not yes_I_have_wrapped_my_main):
      if n_procs>1 and not warning_done:
         print('''
            Warning: You have not wrapped your main routine as follows:
               if __name__=='__main__':
                  MultiProcMagCalc.yes_I_have_wrapped_my_main=True
                  (do stuff)
            This is absolutely necessary on Windows for parallelizing the magnetics calculations.
            (Alternately, you may have forgotten to set MultiProcMagCalc.yes_I_have_wrapped_my_main=True)
            Now dumbing down to single-core computing.
         ''')
         warning_done=True
      for coil in coils:
         Bfunc=getattr(coil,func_name)
         B += Bfunc(r)
   else: 
      elements=coils.flat()
      n_workers=n_procs # min(n_procs, len(elements))
      if MagCalcPool is None:
         logger.debug('n_workers={}, n_procs={}, len(elements)={}'.format(n_workers, n_procs, len(elements)))
         inQ=Manager().Queue()
         outQ=Manager().Queue()
         logQ=Manager().Queue()
         MagCalcPool=Pool(n_workers) 

      for coil in elements:
         try:
            logger.debug('Putting "{}" into queue'.format(coil.name))
         except:
            logger.debug('Putting a current source into queue')
         inQ_put(inQ,coil)
      for n in range(n_workers): #@UnusedVariable
         inQ.put(_MagCalcDone())
      for n in range(n_workers): #@UnusedVariable
         logger.debug('starting a worker')
         MagCalcPool.apply_async(_Bcalc, (r, inQ, outQ, func_name, logQ))
      # MagCalcPool.close()
      for n in range(n_workers): #@UnusedVariable
         B += outQ.get()
   return B
   