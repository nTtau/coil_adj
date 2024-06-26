#Code to test parallelisation of magnetic field code 
#Should speed up magnetic field calculation time allowing for finer sampling meshes to be reasonably used 
#Run each coil as a seperate process? for 12 coils use 12 threads? 
#Uses a filament approach -> user must provide coordinates mapping the centre of the coil 

#from CurrentFilament import CurrentFilament
import numpy as np
from numpy import linspace
from pylab import *
import csv
import multiprocessing as mp
from CurrentFilament import CurrentFilament
from CurrentCollection import CurrentCollection
import time


#print("Number of processors available: ", mp.cpu_count())

#READ IN TF COIL COORDS
def Coord_read(coil_num,coords_dir): 
    
    coil_count = 1
    while coil_count <= coil_num:

        count=0

        #print(coil_count,count)

        file3_name=coords_dir + "/c_" + str(coil_count) + ".csv"
        print(file3_name)
        #print(file3_name)

        TF_coords= open(file3_name, "r")
        file3 = csv.DictReader(TF_coords)

        coil_x_str = []
        coil_y_str = []
        coil_z_str = []
        #print("length : ",len(coil_x_str),len(coil_y_str),len(coil_z_str))

        tot_point=0 

        for col in file3:

                #Read in values
            coil_x_str.append(col['X'])
            coil_y_str.append(col['Y'])
            coil_z_str.append(col['Z'])
            tot_point=tot_point+1
            
        coords = np.zeros((tot_point,3))
        #print("total points= : ", tot_point)

        while count <= tot_point-1: 
      
           # print("count= ",count)
            coords[count,0]=float(coil_x_str[count])
            coords[count,1]=float(coil_y_str[count])
            coords[count,2]=float(coil_z_str[count])

            if count ==0 and coil_count==1: 
 
#                print("0,1 test")
                coords_tot = np.zeros((coil_num,tot_point,3))

            coords_tot[coil_count-1,count,0]=float(coil_x_str[count])
            coords_tot[coil_count-1,count,1]=float(coil_y_str[count])
            coords_tot[coil_count-1,count,2]=float(coil_z_str[count])
            
            count=count+1


 #       print(coords_tot[coil_count-1,:,:])
        coil_count=coil_count+1
    return(coords_tot)
       
#COORDS OF EACH TF COIL CALCULATED
 
#NOW CALCULATE MAGNETIC FIELD FROM EACH COIL      

def B_per_coil(array_in,c_coil_num,coil_num,TF_datafile_path,grid_path):

    #Call code to generate x,y,z grid 

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

    coords_total = int((x_points)*(y_points)*(z_points))
#    print('coords_total =',coords_total)

#    print(x_min,x_max,dx,y_min,y_max,dy,z_min,z_max,dz)

    #Check TF coordinates are correct 

    n = np.size(array_in[:,0]) 
#    print("array_in SIZE: ",np.size(array_in),np.shape(array_in))
#    print("n: ",n,"test:",test)
#
#    for i in range (n): 
 #       print(i,array_in[i,0],array_in[i,1],array_in[i,2])
    
    new= array_in
#    print("array_in SIZE: ",np.size(array_in),np.shape(array_in))
#    print(new)
 #   print("NEW PRINTED")

    #Calculate magnetic field 
# creating empty lists to write to later
    Coil_num_str = []
    N_str = []
    I_str= []

    coil_counter =0

    
    TF_val_file = open(TF_datafile_path,'r')  
    file3 = csv.DictReader(TF_val_file)

    for col in file3:

        #count number of coils
        coil_counter=coil_counter+1

        #Read in values
        Coil_num_str.append(col['coil_num'])
        N_str.append(col['N'])
        I_str.append(col['I (A)'])

    #Create correct size arrays to store info about each coil
    N = np.zeros(coil_num)
    I= np.zeros(coil_num)

# convert to correct type (integer/float)
    count=0

    total_field=np.zeros((coords_total,6))

    while count<= coil_num-1:
   
        N[count]=int(N_str[count]) 
        I[count]=float(I_str[count])
       # print(Coil_num_str[count],N[count],I[count])

        count=count+1

#    print ("All coils identified")
#    print("COIL VALS:", test,N[test],I[test])

   # print(array_in)

    #Coil vals assigned, calculate magnetic field

        #print(N[coil_count-1]*I[coil_count-1])
    print("fil array shape:",np.shape(array_in))
    c1=CurrentFilament(array_in,N[c_coil_num]*I[c_coil_num])
        
        
    l=c1.length()

        #print("Coil, ", coil_count, "length = ",l)
        #print(coil_count)

    centre=np.zeros((1,3))
    point_coords=np.zeros((1,3))
 
    x_current=x_min 
    y_current=y_min
    z_current=z_min

    point_num=0

    field_array=np.zeros((coords_total,6))
 #   print("Calculating field array values")
 
    while x_current <= (x_max+0.5*dx):

        y_current = y_min

        while y_current <= (y_max+0.5*dy):

            z_current = z_min
        
            while z_current <= (z_max+0.5*dz): 
#                print(x_current,y_current,z_current)

                           # print(point_num)

                point_coords[0,0]=centre[0,0]+x_current
                point_coords[0,1]=centre[0,1]+y_current
                point_coords[0,2]=centre[0,2]+z_current
                  
                Bnet=c1.B(point_coords)
                                #print("coords and field") 
                                #print(point_coords[0,0],point_coords[0,1],point_coords[0,2],Bnet[0,0],Bnet[0,1],Bnet[0,2])

                field_array[point_num,0] = point_coords[0,0]
                field_array[point_num,1] = point_coords[0,1]
                field_array[point_num,2] = point_coords[0,2]
                field_array[point_num,3] = Bnet[0,0]
                field_array[point_num,4] = Bnet[0,1]
                field_array[point_num,5] = Bnet[0,2]

                point_num=point_num+1
                z_current = z_current+dz


            y_current=y_current + dy

        x_current = x_current + dx

        #Coordinates

#        total_field[:,0] = field_array[:,0]
 #       total_field[:,1] = field_array[:,1]
  #      total_field[:,2] = field_array[:,2]

        #Magnetic field 
   #     total_field[:,3] = total_field[:,3]+field_array[:,3]
    #    total_field[:,4] = total_field[:,4]+field_array[:,4]
     #   total_field[:,5] = total_field[:,5]+field_array[:,5]

#        print("Coil = ",coil_count,field_array[coords_total-1,:])
 #       print("Total so far : Coil = ",coil_count,total_field[coords_total-1,:])

      #  print(total_field[coords_total-1,:])
        #print(N[coil_count-1]*I[coil_count-1])

    
    return (field_array)

#Parallel version of TF field code 
#Calculate B field at all points for a single coil in parallel  
#Add together in series 

def TF_FIELD(coil_num,proc_num,TF_data,coords_dir,grid_path):

    start_t=time.perf_counter()

#    TF_datafile_path =  "Example_files/TF_vals.csv"
#    coords_dir =  "new_TF"
 #   coil_num=12
  #  proc_num = 12 #Number of processors to use
 #   print(coil_num, proc_num,coords_dir)
 #   print(TF_data)

#READ TF COORDS INTO AN ARRAY
 #   print("coord_ar=Coord_read(",coil_num,TF_data,coords_dir)
    coord_ar=Coord_read(coil_num,coords_dir)

#SPLIT COIL COORD ARRAY INTO ONE "CHUNK" PER COIL
    chunks = [coord_ar[i] for i in range(coil_num)] 

#Make pool
    pool = mp.Pool(proc_num)

#Compute B field from each coil
    async_result = [ pool.apply_async(B_per_coil, (chunks[i],i,coil_num,TF_data,grid_path)) for i in range(coil_num)]

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
    B_file= open("fil_coil_B.csv", "w")
    print("x,y,z,Bx,By,Bz",file=B_file)

    point_num=np.size(total[:,0])

    for i in range(point_num): 

#    Bmag = np.sqrt((total[i,3]**2) +(total[i,4]**2)+(total[i,5]**2))
        print(total[i,0],",",total[i,1],",",total[i,2],",",total[i,3],",",total[i,4],",",total[i,5],file=B_file)


#Calculate total time taken
    end_t=time.perf_counter()
    tot_t = end_t-start_t#

#Print total time
    print("Total time for coils:",tot_t)

    return(total)

#out=TF_FIELD(12,12,"Example_files/TF_vals.csv","new_TF")
#out=TF_FIELD(12,12,"Example_files/TF_vals.csv","new_TF") 



