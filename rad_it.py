#Code to iterate through changing coil radii -> modify currrent until target B centre field hit, calculate hoop stress 
#Output number of coils/spacing between coils, coil radius,coil current,Bcentre,coil dr/dz and hoop stress as csv -> maybe do average,min,max hoop stress?

# Use csv for NN training/M.O.O
# Train simple neural network? -> Then do M.O.O to optimise for min hoop stress and cost 
# Could we also add some sort of field uniformity measure? then make that another to optimise 
# Higher weighting to field uniformity -> higher weighting to physics considerations? 
# Higher weighting to min hoop stress -> higher weighting to engineering considerations? 
# Higher weighting to min cost -> higher weighting to cost considerations 


#Would be valid for any solenoid set up? would have to assume evenly spaced coils  
#Run each coil as a seperate process? for 16 coils use 16 threads? 

#from CurrentFilament import CurrentFilament
import numpy as np
from numpy import linspace
from pylab import *
import csv
import multiprocessing as mp
from CurrentFilament import CurrentFilament
from CurrentCollection import CurrentCollection
import time
from MagCoil import MagCoil
import current_est as c_e
import parallel_force_calc as forces

u0 = (np.pi)*4*(10**-7) # mag const


#print("Number of processors available: ", mp.cpu_count())

#Creates a solenoid with the central axis in z direction 
#Bc = target central field

#Central cell of a magnetic mirror acts like a solenoid 
#Inputs are number of coils(n_coils), target field at centre(Bc), coil average radius (r), solenoid length(l), max allowed current per turn(max_c_turn) 
#For now use max current load ratings at room temp -> add in adjustment for larger temps later, we will assume we can add sufficient cooling to maintain this temp 
#Use AWG guide here:https://www.engineeringtoolbox.com/wire-gauges-d_419.html for copper coil 
#Need to find same for YBCO/REBCO HTS 

def gen_so_coil_array(n_coils,Bc,r,l,max_c_turn,turn_dr,turn_dz): 

   #Initial guess for current per coil: 
    #coil z spacing
    coil_z_s = l/n_coils
    I=c_e.sol_I_calc(n_coils,l,Bc)
    nturns = I/max_c_turn 
#    print(nturns)
    nz_float= (np.sqrt(nturns)) 
    nz_int = int(nz_float)

    #Round number of z turns up if non integer
    if nz_float - nz_int > 0.000001: 
        nz = nz_int + 1 
    else: 
        nz=nz_int

    nr = nz 
    N_tot = nr*nz 
    I_per_turn = I/N_tot

    dr=turn_dr*nr
    dz=turn_dz*nz

    coil_ar = np.zeros((n_coils,12))

    for c_n in range(n_coils):
        #Set coil z spacing 
        coil_z = -l/2 + (0.5*coil_z_s) + (c_n*coil_z_s) 

        #Add values to the data array 
        coil_ar[c_n,0] = nr 
        coil_ar[c_n,1] = nz
        coil_ar[c_n,2] = I_per_turn  
        coil_ar[c_n,3] = r
        coil_ar[c_n,4] = dr
        coil_ar[c_n,5] = dz  
        coil_ar[c_n,6] = 0.0
        coil_ar[c_n,7] = 0.0
        coil_ar[c_n,8] = coil_z
        coil_ar[c_n,9] = 0.0
        coil_ar[c_n,10] = 0.0
        coil_ar[c_n,11] = 1.0

    print("number of coils:",n_coils,"coil average radius:",r,"dr:",dr,"dz:",dz)
    print("Current needed per coil:",I)
  #Initial coil set up defined 
    return(coil_ar) 


#Function to determine the coordinates of sampling points down the solenoid central axis ( assumes central axis in z direction for now) 
# Generates points on a line equal length to that of the solenoid 

def centre_sample_points(npoints,l):  

    spacing= l/npoints
    line_coords = np.zeros((npoints,3))
#    print(np.shape(line_coords))
    

    for i in range(npoints):

 #       print(i)
        z_pos = (-l/2) + (i* spacing)
        line_coords[i,0] = 0.0
        line_coords[i,1] = 0.0 
        line_coords[i,2] = z_pos 


    return(line_coords)


def B_central(coil_array,line_coords):

    npoints = np.size(line_coords[:,0])
    coil_num = np.size(coil_array[:,0])
 #   print("number of coils:",coil_num)
#    print("number of sampling points",npoints)

    B_calc=np.array((npoints,6))

    #Iterate through sample coordinates 
    for i in range(npoints):

        x = line_coords[i,0] 
        y = line_coords[i,1] 
        z = line_coords[i,2]  

        point=np.zeros((1,3))

        point[0,0]=x
        point[0,1]=y
        point[0,2]=z

        #Loop through coils in the array 
        for c_n in range(coil_num): 

            cc =MagCoil(array([coil_array[c_n,6],coil_array[c_n,7],coil_array[c_n,8]]), array([coil_array[c_n,9],coil_array[c_n,10],coil_array[c_n,11]]), R=coil_array[c_n,3], I=coil_array[c_n,0]*coil_array[c_n,1]*(coil_array[c_n,2]))
 
            B= cc.B(point)
            if c_n ==0: 
                Btot = B 
            else: 
                #sum magnetic fields
                Btot = Btot + B 

#            print("B:",B)
 #       print("Btot:",Btot)

        #Add to array of coordinates and corresponding field 

        output = np.array([x,y,z,Btot[0,0],Btot[0,1],Btot[0,2]]) 

        if i==0:

            final_out = output

        else:

            final_out = np.vstack((final_out,output))

      

    return(final_out) 

#Calculate mean B, range, var of mag field for sampled points 
def stats_B(B_ar): 

    npoints=np.size(B_ar[:,0])
    for i in range(npoints): 
        if i ==0:
            sum_Bx =  B_ar[i,3]
            sum_By =  B_ar[i,4]
            sum_Bz =  B_ar[i,5]
        else:
            sum_Bx = sum_Bx + B_ar[i,3]
            sum_By = sum_By + B_ar[i,4]
            sum_Bz = sum_Bz + B_ar[i,5]

    mean_Bx = sum_Bx/npoints
    mean_By = sum_By/npoints
    mean_Bz = sum_Bz/npoints
    mean_B = np.array([mean_Bx,mean_By,mean_Bz])

    #Measure variance of field  
    #Gives an idea of field uniformity -> lower standard deviation is better 

    for i in range(npoints): 
        if i ==0:
            sum_vals = ((mean_B - np.array([B_ar[i,3],B_ar[i,4],B_ar[i,5]])))**2
        else:
            sum_vals = sum_vals + ((mean_B - np.array([B_ar[i,3],B_ar[i,4],B_ar[i,5]])))**2

    #    print(i,(mean_B - np.array([B_ar[i,3],B_ar[i,4],B_ar[i,5]])),sum_vals)
        
    var = sum_vals/npoints 


   #Identify min and max values 

 
    min_Bx = min(B_ar[:,3]) 
    max_Bx = max(B_ar[:,3])

    min_By = min(B_ar[:,4]) 
    max_By = max(B_ar[:,4])

    min_Bz = min(B_ar[:,5]) 
    max_Bz = max(B_ar[:,5])

   #Range of values for Bx,By,Bz across sampling coordinates
    range_B=np.array([max_Bx-min_Bx,max_By-min_By,max_Bz-min_Bz])
    #min and max
    min_field=np.array([min_Bx,min_By,min_Bz])
    max_field=np.array([max_Bx,max_By,max_Bz])

    return(mean_B,max_field,min_field,range_B,var)

#Calculates the total volumes of the coils given above 

def pancake_coil_vol(coil_data): 

    c_n = np.size(coil_data[:,0])
    volumes=np.zeros((c_n,1))
    for i in range(c_n): 
        r_av = coil_data[i,3]
        dr = coil_data[i,4]
        dz = coil_data[i,5]
        rmax = r_av + (dr/2) 
        rmin = r_av - (dr/2) 
        vout = np.pi*(rmax**2)*dz 
        vin = np.pi*(rmin**2)*dz
#        print(rmax,rmin)
 #       print(vout,vin) 
        
        volumes[i,0] = vout-vin

    return(volumes)

#INPUTS 
samp_points=1000
sol_l = 92.0 
B_targ = 3.0
#Use 400 AWG Cu for now
max_c_t = 415.0
#Ensure measurements in metres
t_dr=13.34 * (10**-3)
t_dz=13.34 * (10**-3)
radius = 3.0

cn_scan = open("cn_scan.csv", 'w')

print("#coils,mean field(T),range(T),var(T),av hoop stress per coil,total volume of material ",file=cn_scan)

# RUN CODE THROUGH
for coil_n in range(2,46,2): 

    centre_coords=centre_sample_points(samp_points,sol_l)
    coil_info=gen_so_coil_array(coil_n,B_targ,radius,sol_l,max_c_t,t_dr,t_dz)
#print("COIL INFO:") 
#print(coil_info)
    B_out=B_central(coil_info,centre_coords)
    Nr = coil_info[0,0]
    Nz = coil_info[0,1]
#print(np.shape(B_out))
    print("Coils used:", coil_n) 
    mean,max_f,min_f,rang,var=stats_B(B_out)
    print("mean field:",mean)
    print("max field values:",max_f,"min_field values:",min_f)
    print("total range:",rang)
    print("variance of field",var)
    vols=pancake_coil_vol(coil_info)
    vol_tot = 0.0
    for i in range((np.size(vols[:,0]))): 
        vol_tot = vol_tot + vols[i,0]
    print("Total coil volume needed:", vol_tot)
#Output coordinates and net field 

#PERFORM FORCE/STRESS CALCULATION
    proc_num = 12 # Number of processors
    nfil=100 #Number of filaments to split each coil into 
    B_back = [0.0,0.0,0.0] #Background magnetic field (in T) - only valid for uniform background fields


    #Calculate forces on circular coils: 
    coil_num = np.size(coil_info[:,0]) #
    print("Number of circular coils:",coil_num)
# all_coils(proc_num,nfil,B_back,use_circ_c=False,data_array=[0,0,0],use_fil_c=False,fil_ar_in=[0,0,[0,0,0]]):
    hs_tot=forces.all_coils(proc_num,nfil,B_back,True,coil_info,False)
    hs_av = hs_tot*(10**-6)/coil_num

    print(coil_n,mean[2],rang[2],var[2],hs_av,vol_tot,Nr,Nz,file=cn_scan) 

    print("total hoop stress:",hs_tot*(10**-6),"MPa")
    print("Average hoop stress per coil:", hs_tot*(10**-6)/coil_num,"MPa")


#print(coil_info)








