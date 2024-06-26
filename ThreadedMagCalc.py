'''
Created on Oct 1, 2012

@author: sieck
'''
import threading
import Queue

verbose=False
n_threads=6

class MagCalcDone():
   pass

class MagCalcThread(threading.Thread):
   def __init__(self, inQ, outQ, r, threadnum):
      threading.Thread.__init__(self)
      self.inQ=inQ
      self.outQ=outQ
      self.r=r
      self.threadnum=threadnum

   def run(self):
      while True:
         coilnum,coil=self.inQ.get()
         if verbose:
            print 'Thread {} is calculating B for object {}'.format(self.threadnum, coilnum)
         self.outQ.put((self.threadnum,coil.B(self.r)))
         self.inQ.task_done()

class MagCalcCollectorThread(threading.Thread):
   def __init__(self, inQ, outQ):
      threading.Thread.__init__(self)
      self.inQ=inQ
      self.outQ=outQ

   def run(self):
      n,B=self.inQ.get()
      if verbose:
         print 'Starting with B field from thread {}'.format(n)
      while True:
         n,newB=self.inQ.get()
         if isinstance(newB,MagCalcDone):
            if verbose:
               print 'Collector reached end flag, posting result'
            self.outQ.put(B)
            return
         if verbose:
            print 'Adding B field from thread {}'.format(n)
         B += newB
         self.inQ.task_done()

def ThreadedMagCalc(coils,r,B0=0):
   inQ=Queue.Queue()
   outQ=Queue.Queue()
   outQ.put((0,B0))
   outQ2=Queue.Queue()
   for n in range(n_threads):
      thread=MagCalcThread(inQ,outQ,r, n+1)
      thread.setDaemon(True)
      thread.start()
   for job in enumerate(coils):
      inQ.put(job)
   thread=MagCalcCollectorThread(outQ,outQ2)
   thread.start()
   inQ.join()
   outQ.put((0,MagCalcDone()))
   return outQ2.get()
   