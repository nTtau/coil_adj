
import numpy as np
from pylab import *
import csv


#Reads in bounding box file and generates the min and max x,y,z for a cubiodal grid slightly larger than the gemometry 
# Outputs file named grid_vals.csv that can be used by magnetic field code to generate the sample points of the grid 

def grid_gen(bound_b_path,dx,dy,dz):


#Set total number of coils used

    #READ BOOUNDING BOX FILE 

    val_file =open("grid_vals.csv", 'w')

    bound_f= open(bound_b_path, "r")
    file3 = csv.DictReader(bound_f)


    for col in file3:

        #Read in values
        axis= col['Axis']
        if axis =='X': 
            print("x axis")
            xmax = float(col['Max'])
            xmin = float(col['Min'])
        if axis =='Y': 
            print("y axis")
            ymax = float(col['Max'])
            ymin = float(col['Min'])
        if axis =='Z': 
            print("z axis")
            zmax = float(col['Max'])
            zmin = float(col['Min'])

    #Extend grid so integer number of points in x,y,z -> needs to be integer for grid

    nx = int((xmax-xmin)/dx) +2
    x_length = nx*dx 
    ny = int((ymax-ymin)/dy) +2
    y_length = ny*dy
    nz = int((zmax-zmin)/dz) +2
    z_length = nz*dz

    x_max_final = (x_length/2)
    x_min_final = -(x_length/2)
    y_max_final =  (y_length/2)
    y_min_final = -(y_length/2)
    z_max_final =  (z_length/2)
    z_min_final = -(z_length/2)
      
 #   print ( "Final range: ", x_max_final, x_min_final,y_max_final, y_min_final,z_max_final, z_min_final)  

    print("x_min,x_max,dx,y_min,y_max,dy,z_min,z_max,dz" , file=val_file)
    print (x_min_final,",",x_max_final,",",dx,",",y_min_final,",",y_max_final,",",dy,",",z_min_final,",",z_max_final,",",dz,file=val_file)
    
    #x_min_str,x_max_str,dx_str,y_min_str,y_max_str,dy_str,z_min_str,z_max_str,dz_str = row[:9]



 









