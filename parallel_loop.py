#Code to test parallelisation of magnetic field code 
#Should speed up magnetic field calculation time allowing for finer sampling meshes to be reasonably used 
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


#print("Number of processors available: ", mp.cpu_count())

#NOW CALCULATE MAGNETIC FIELD FROM EACH COIL      

def B_per_loop(c_coil_num,coil_num,coil_data,grid_path):


#    print("CALC FOR CIRC COIL: ",c_coil_num)
#Set grid params: 

    grid_file = open(grid_path,'r')  
    file2 = csv.DictReader(grid_file)

    for row in csv.reader(grid_file):
        x_min_str,x_max_str,dx_str,y_min_str,y_max_str,dy_str,z_min_str,z_max_str,dz_str = row[:9]


    dx=float(dx_str)
    dy=float(dy_str)
    dz=float(dz_str)

    x_min = float(x_min_str) 
    x_max = float(x_max_str)
    y_min = float(y_min_str)
    y_max = float(y_max_str)
    z_min = float(z_min_str)
    z_max = float(z_max_str)

    x_points=1+((x_max-x_min)/dx)
    y_points=1+((y_max-y_min)/dy)
    z_points=1+((z_max-z_min)/dz)
    print("point numbers:",x_points,y_points,z_points)
    print("dx,dy,dz:", dx,dy,dz)
    coords_total = int((x_points)*(y_points)*(z_points))
#    print('coords_total =',coords_total)


# Calculates the magnetic field from one coil using its properties

    #Coil properties from coil_data file 
    #READ COIL DATA FOR CORRECT COIL 
    
    #Open coil_data_file 
    #Read line corresponding to c_coil_num 

    coil_counter =0

    dat=[]

    filename = open(coil_data, 'r')
    file = csv.DictReader(filename)

    for col in file:

        #count number of coils
        coil_counter=coil_counter+1

        if coil_counter ==c_coil_num+1: 
#            print(c_coil_num)
            

        #Read in values
#            print(coil_counter)
            Nr=int(col['R_turns'])
            Nz=int(col['Z_turns'])
            I_val=float(col['I (A)'])
            r=float(col['R_av'])
            dr=float(col['dr'])
            coil_dz=float(col['dr'])
            x_c=float(col['Coil_X'])
            y_c=float(col['Coil_Y'])
            z_c=float(col['Coil_Z'])
            x_n=float(col['Normal_x'])
            y_n=float(col['Normal_y'])
            z_n=float(col['Normal_z'])

            N_val = Nr*Nz


    #Main script to run B calc for given coil 

#    print(c_coil_num, I_val)
    coil_centre = array([x_c,y_c,z_c]) # coil centre 
    coil_normal = array([x_n,y_n,z_n]) # normal vector to plane of coil 


   # print(coil_centre) 
   # print(coil_normal)

        #Define coil 
    c1=MagCoil(coil_centre, coil_normal, R=r, I=N_val*I_val) 

        #Use to define centre point of grid 
    centre=MagCoil(array([0.0,0.0,0.0]), array([0,0,1.0]), R=r, I=N_val*I_val)
 
    x_current=x_min 
    y_current=y_min
    z_current=z_min

    point_num=0

    field_array=np.zeros((coords_total,6))
 
    while z_current <= (z_max+0.5*dz):

        y_current = y_min

        while y_current <= (y_max+0.5*dy):

            x_current = x_min
        
            while x_current <= (x_max+0.5*dx): 

                           # print(point_num)

                smidge =array([x_current,y_current,z_current])
                Bnet=c1.B(centre.r0+smidge)#+c2.B(c3.r0+smidge)
                Bx = Bnet[0,0] 
                By = Bnet[0,1]
                Bz = Bnet[0,2]

                #Store values to an array 
                        
                field_array[point_num,0] = x_current
                field_array[point_num,1] = y_current
                field_array[point_num,2] = z_current
                field_array[point_num,3] = Bx
                field_array[point_num,4] = By
                field_array[point_num,5] = Bz
                                #field_array[point_num,6] = B_net_mag]                   

                point_num=point_num+1

                x_current = x_current+dx

            y_current=y_current+dy

        z_current = z_current + dz

    return(field_array)

#Parallel version of TF field code 
#Calculate B field at all points for a single coil in parallel  
#Add together in series 

def PF_FIELD(coil_num,proc_num,PF_data,grid_path):

    start_t=time.perf_counter()

#    TF_datafile_path =  "Example_files/TF_vals.csv"
#    TF_coords_dir =  "new_TF"
 #   coil_num=12
  #  proc_num = 12 #Number of processors to use
 #   print(coil_num, proc_num)
#    print(PF_data)

#SPLIT COIL COORD ARRAY INTO ONE "CHUNK" PER COIL

#Make pool
    pool = mp.Pool(proc_num)

#Compute B field from each coil
    async_result = [ pool.apply_async(B_per_loop, (i,coil_num,PF_data,grid_path)) for i in range(coil_num)]

#Close pool and join results
    pool.close()
    pool.join()

#Sum all magnetic field values 
    for i in range(coil_num): 

        if i ==0:
            total = async_result[i].get()
        else: 
            coil_B = async_result[i].get()
#        print(np.shape(coil_B))
         #Don't change x,y,z coords
 #           total[:,0] = coil_B[:,0]
  #          total[:,1] = coil_B[:,1]
   #         total[:,2] = coil_B[:,2]
         #Sum magnetic field components 
            total[:,3] = total[:,3]+coil_B[:,3]
            total[:,4] = total[:,4]+coil_B[:,4]
            total[:,5] = total[:,5]+coil_B[:,5]

#    print("RUN COMPLETE")

#print total field to file 
    B_file= open("PF_B.csv", "w")
    print("x,y,z,Bx,By,Bz",file=B_file)

    point_num=np.size(total[:,0])

    for i in range(point_num): 

#    Bmag = np.sqrt((total[i,3]**2) +(total[i,4]**2)+(total[i,5]**2))
        print(total[i,0],",",total[i,1],",",total[i,2],",",total[i,3],",",total[i,4],",",total[i,5],file=B_file)


#Calculate total time taken
    end_t=time.perf_counter()
    tot_t = end_t-start_t#

#Print total time
    print("Total time for circ coils:",tot_t)

    return(total)

#out=PF_FIELD(16,16,"Example_files/pf_example.csv","grid_vals.csv")
#out=TF_FIELD(12,12,"Example_files/TF_vals.csv","new_TF") 


#out=TF_FIELD(12,12,"Example_files/TF_vals.csv","new_TF")
#out=TF_FIELD(12,12,"Example_files/TF_vals.csv","new_TF") 
#b=B_per_coil(1,12,"Example_files/pf_example.csv")
#print(b)



