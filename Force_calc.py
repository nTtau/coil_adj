
import numpy as np
from numpy import linspace
from pylab import *
import multiprocessing as mp
import parallel_force_calc as forces


proc_num = 2 # Number of processors
nfil=100 #Number of filaments to split each coil into 
B_back = [0,0,0] #Background magnetic field (in T) - only valid for uniform background fields


#Calculate forces on circular coils: 
circ_dat=forces.coil_file_reader("Example_files/Helmholtz_ex.csv")
coil_num = np.size(circ_dat[:,0]) 
print("Number of circular coils:",coil_num)
# all_coils(proc_num,nfil,B_back,use_circ_c=False,data_array=[0,0,0],use_fil_c=False,fil_ar_in=[0,0,[0,0,0]]):
hs_tot=forces.all_coils(proc_num,nfil,B_back,True,circ_dat,False)

print("total hoop stress:",hs_tot*(10**-6),"MPa")
print("Average hoop stress:", hs_tot*(10**-6)/coil_num,"MPa")











