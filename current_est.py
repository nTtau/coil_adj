
import numpy as np
from pylab import *
import csv

#SCRIPT TO ESTIMATE THE CURRENT IN A COIL/SOLENOID TURN GIVEN A TARGET B CENTRE 
#USE AS A STARTING POINT FOR DETERMINING COIL CURRENTS

u0 = (np.pi)*4*(10**-7) # mag const

#Reads in bounding box file and generates the min and max x,y,z for a cubiodal grid slightly larger than the gemometry 
# Outputs file named grid_vals.csv that can be used by magnetic field code to generate the sample points of the grid 

#generate solenoid with central axis in z direction
def sol_gen(N,L,R,B_c_t,dr,dz):


#Set total number of coils used

    #READ BOUNDING BOX FILE 

    I=sol_I_calc(N,L,B_c_t)

    val_file =open("solenoid.csv", 'w')


    print("R_turns,Z_turns,I (A),R_av,dr,dz,Coil_X,Coil_Y,Coil_Z,Normal_x,Normal_y,Normal_z" , file=val_file)

    z_ar = np.linspace(-(L/2),(L/2),num=N,endpoint=True)
#    print(z_ar)
    for p in range(N):


        Coil_X = 0.0 
        Coil_Y = 0.0
        Coil_Z = z_ar[p]

        print (1,1,I,R,dr,dz,Coil_X,Coil_Y,Coil_Z,0,0,1,sep=",",file=val_file)

    return()
    #x_min_str,x_max_str,dx_str,y_min_str,y_max_str,dy_str,z_min_str,z_max_str,dz_str = row[:9]

#B_c_t = target val of central field 
def sol_I_calc(N,L,B_c_t): 

    #B = u0 NI /L 
    #NI = BL/u0 
    #I = BL/u0N 
    I_val=(B_c_t*L)/(u0*N)
    #I_val = 3238500.0/N
#    I_val=1000
 
    return(I_val)
    
def coil_I_calc(B_c_t,R): 

    I_val = (2*B_c_t*R)/(u0)
    return(I_val)
 

#I = sol_I_calc(88,1.5,2.71)
#print(I)

#I = coil_I_calc(3.0,2.0)
#print(I)

#Starting estimate of current in each solenoid turn/coil 
Nturns=88
Length = 1.5
sol_gen(Nturns,Length,0.13,2.71,0.017,(Length/Nturns))

#sol_gen(2,0.05,0.025,2.0,0.01,0.05/2)





