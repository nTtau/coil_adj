from MagCoil import MagCoil
from CurrentFilament import CurrentFilament
import numpy as np
from numpy import linspace
from pylab import *
from mayavi import mlab as MLab
from pylab import *
from CurrentCollection import CurrentCollection
from Meshtricate import Meshtricate
import MultiProcMagCalc
import multiprocessing as mp
import multiprocessing
import logging
import csv


#Code to sum magnetic field of n arrays
def field_sum(ar_list): 

    n=len((ar_list))

    #Sum field components 

    for i in range (n): 

        if i ==0: 
            tot_field = ar_list[i] 

        else: 
            new_field = ar_list[i]

            #Sum fields, dont change coordinates 

            tot_field[:,3] = tot_field[:,3] + new_field[:,3] 
            tot_field[:,4] = tot_field[:,4] + new_field[:,4] 
            tot_field[:,5] = tot_field[:,5] + new_field[:,5] 


    #Check array size
    points= np.size(tot_field[:,0] )
  #  print(points)
    FieldMag = np.zeros((points,1))#

    for i in range(points): 

        FieldMag[i,0] = np.sqrt((tot_field[i,3]**2) + (tot_field[i,4]**2) +(tot_field[i,5]**2))


    #Calculate BMAG

    tot_field=np.hstack((tot_field,FieldMag))

    tot_file =open("B_tot.csv", 'w')
    print("x,y,z,Bx,By,Bz,Bmag",file=tot_file)

    for i in range(points): 

        print(tot_field[i,0],",",tot_field[i,1],",",tot_field[i,2],",",tot_field[i,3],",",tot_field[i,4],",",tot_field[i,5],",",tot_field[i,6],file=tot_file)

    return(tot_field)



#IGNORE FOR NOW, NEEDS REWRITING FOR STELLARATORS 

def field_map(geom_type,coil_data_path,TF_coords_dir):
    #Set file names to write values to 
    start_time = time.perf_counter()

    #total_file=open('B_plot_multi.csv', 'w')
    #Bx_file=open('B_x_multi.csv', 'w')
    #By_file=open('B_y_multi.csv', 'w')
    #Bz_file=open('B_z_multi.csv', 'w')

#Set total number of coils used

#Read in coil specs 

    if geom_type =="Cylinder": 

        B_file= open("B_multi.csv", "r")
        filename = open(coil_data_path, 'r')
        file = csv.DictReader(filename)

    elif geom_type =="Torus": 

        
        PF_field_file =open("PF_B.csv", 'r')

        PF_geom_file =open(coil_data_path, 'r')
        file2 = csv.DictReader(PF_geom_file)
        
        TF_field_file =open("TF_B.csv", 'r')

        tot_file =open("B_tot.csv", 'w')
        print("x,y,z,Bx,By,Bz,Bmag",file=tot_file)

        PF_array,PF_points=B_csv_read(PF_field_file)
       # print(PF_array)

        TF_array,TF_points=B_csv_read(TF_field_file)

        if TF_points==PF_points: 
            print("Correct number of points in each file")

            Total_B=np.zeros((PF_points,7))
 
            Total_B[:,0] = PF_array[:,0]
            Total_B[:,1] = PF_array[:,1]
            Total_B[:,2] = PF_array[:,2]
            Total_B[:,3] = PF_array[:,3] + TF_array[:,3]
            Total_B[:,4] = PF_array[:,4] + TF_array[:,4]
            Total_B[:,5] = PF_array[:,5] + TF_array[:,5]

         #   Total_B[:,3] = TF_array[:,3]
          #  Total_B[:,4] = TF_array[:,4]
           # Total_B[:,5] = TF_array[:,5]

            Total_B[:,6] = sqrt((Total_B[:,3]**2)+(Total_B[:,4]**2)+(Total_B[:,5]**2))

            count=0
            while count <= PF_points-1: 

                print(Total_B[count,0],",",Total_B[count,1],",",Total_B[count,2],",",Total_B[count,3],",",Total_B[count,4],",",Total_B[count,5],",",Total_B[count,6],file=tot_file)
                count=count+1
 
            print("Total field calculated")

            print("Plotting field map in Mayavi")


            Circ_Coil_render("Torus",coil_data_path)
            TF_render(12,TF_coords_dir)

        #Add moveable planes to show the field

#Creates grid of x_points by y points by z points but as 3d arrays (format needed for mayavi same coordinates as the previous grid generator)


            grid_file = open("grid_vals.csv",'r')  
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

            x,y,z=mgrid [x_min:x_max+dx:dx,y_min:y_max+dy:dy,z_min:z_max+dz:dz ]

#reshape arrays into format mayavi needs

            Bx=Total_B[:,3].reshape(x.shape) 
            By=Total_B[:,4].reshape(y.shape)
            Bz=Total_B[:,5].reshape(z.shape)
            B_mag_reshape=Total_B[:,6].reshape(x.shape)

            MLab.flow(x,y,z,Bx,By,Bz,seedtype='plane',integration_direction='both',seed_scale=2.0,seed_resolution=5,seed_visible=False,color=(1,1,1))
            x_plane=MLab.volume_slice(x,y,z,B_mag_reshape,plane_orientation='x_axes',colormap='blue-red') #vmin=0,vmax=1, figure=B_mag_fig)
            y_plane=MLab.volume_slice(x,y,z,B_mag_reshape,plane_orientation='y_axes',colormap='blue-red') #vmin=0,vmax=1, figure=B_mag_fig)
            z_plane=MLab.volume_slice(x,y,z,B_mag_reshape,plane_orientation='z_axes',colormap='blue-red') #vmin=0,vmax=5, figure=B_mag_fig)
            MLab.colorbar(object=x_plane,title="BMag",orientation='vertical')
            MLab.draw()
#            MLab.savefig(filename='test.png')
            MLab.show()

        else: 
            print("Incorrect number of points - please rerun the code")


    else:
        print("invalid geometry type given") 

#Add up the magnetic fields 

# creating empty lists to write to later

def B_csv_read(filename):

    file_B = csv.DictReader(filename)

    #x,y,z,Bx,By,Bz

    x_str = []
    y_str= []
    z_str = []
    Bx_str = []
    By_str = []
    Bz_str = []

    total_point_count =0
    for col in file_B:

        #count number of coils
        

        #Read in values
        x_str.append(col['x'])
        y_str.append(col['y'])
        z_str.append(col['z'])
        Bx_str.append(col['Bx'])
        By_str.append(col['By'])
        Bz_str.append(col['Bz'])

  
        total_point_count=total_point_count+1

   # print(total_point_count)
    #Create correct size arrays to store info 

    count = 0 
    Total_array = np.zeros((total_point_count,6))

    while count <= total_point_count-1:

        #print(count,total_point_count)
   
        Total_array[count,0] = float(x_str[count])
        Total_array[count,1] = float(y_str[count]) 
        Total_array[count,2] = float(z_str[count]) 
        Total_array[count,3] = float(Bx_str[count]) 
        Total_array[count,4] = float(By_str[count]) 
        Total_array[count,5] = float(Bz_str[count]) 
       
        count=count+1

    return(Total_array,total_point_count)
 
    
def TF_render(coil_num,coil_coord_dir):

    coil_count=1 
    while coil_count <= coil_num:

        x_coord_str = []
        y_coord_str=[]
        z_coord_str=[]

        geom_file=open(coil_coord_dir + "/TF_" + str(coil_count) + ".csv", "r")
        file1=csv.DictReader(geom_file)

        point_count=0

        for col in file1:

            x_coord_str.append(col['x'])
            y_coord_str.append(col['y'])
            z_coord_str.append(col['z'])
              
            point_count=point_count+1

        points=point_count 

        coords = np.zeros((point_count,3))

        count=0
        while count<= point_count-1:

            coords[count,0]=float(x_coord_str[count])
            coords[count,1]=float(y_coord_str[count])
            coords[count,2]=float(z_coord_str[count])

            count=count+1

        print(coil_count)
        c1=CurrentFilament(coords,point_count)
        c1.render(color=(0,0,0),tube_radius=0.1)

        coil_count=coil_count+1        

   # MLab.draw()
    #MLab.show()


    return()

def Circ_Coil_render(geom_type,coil_path):

# creating empty lists to write to later
    Nr_str = []
    Nz_str = []
    I_str= []
    r_str = []
    dr_str = []
    coil_dz_str = []
    x_centre_str = []
    y_centre_str = []
    z_centre_str = []
    x_normal_str = []
    y_normal_str = []
    z_normal_str = []
#OLD

    coil_counter=0

    if geom_type =="Torus":

        geom_file=open(coil_path, "r")

    elif geom_type =="Cylinder":

        geom_file=open(coil_path, "r")

    file=csv.DictReader(geom_file)
    coil_counter =0

    for col in file:

        #count number of coils
        coil_counter=coil_counter+1

        #Read in values
        Nr_str.append(col['R_turns'])
        Nz_str.append(col['R_turns'])
        I_str.append(col['I (A)'])
        r_str.append(col['R_av'])
        dr_str.append(col['dr'])
        coil_dz_str.append(col['dr'])
        x_centre_str.append(col['Coil_X'])
        y_centre_str.append(col['Coil_Y'])
        z_centre_str.append(col['Coil_Z'])
        x_normal_str.append(col['Normal_z'])
        y_normal_str.append(col['Normal_y'])
        z_normal_str.append(col['Normal_x'])

    coil_num=coil_counter
    print("number of coils = ", coil_num)
    count=0

    #Create correct size arrays to store info about each coil
    Nr = np.zeros(coil_num)
    Nz = np.zeros(coil_num)
    N = np.zeros(coil_num)
    I= np.zeros(coil_num)
    r = np.zeros(coil_num)
    dr = np.zeros(coil_num)
    coil_dz = np.zeros(coil_num)
    x_centre = np.zeros(coil_num)
    y_centre = np.zeros(coil_num)
    z_centre = np.zeros(coil_num)
    x_normal = np.zeros(coil_num)
    y_normal = np.zeros(coil_num)
    z_normal = np.zeros(coil_num)

# convert to correct type (integer/float)

    while count<= coil_num-1:
   
        Nr[count]=int(Nr_str[count]) 
        Nz[count]=int(Nz_str[count]) 
        N[count]=int((Nz[count]*Nr[count])) 
        I[count]=float(I_str[count])
        r[count]=float(r_str[count])
        dr[count]=float(dr_str[count])
        coil_dz[count]=float(coil_dz_str[count])
        x_centre[count]=float(x_centre_str[count])
        y_centre[count]=float(y_centre_str[count])
        z_centre[count]=float(z_centre_str[count])
        x_normal[count]=float(x_normal_str[count])
        y_normal[count]=float(y_normal_str[count])
        z_normal[count]=float(z_normal_str[count])

 #   print(N[count],I[count],N[count]*I[count])

        count=count+1

    print ("All coils identified")

    count=0
    #Add coils to Figure 
    while count <= coil_num-1:
 
        zone=MagCoil([x_centre[count],y_centre[count],z_centre[count]],[x_normal[count],y_normal[count],z_normal[count]],r[count],r_minor=(0.25*dr[count]))
#                print(count, r_outer[count],r_inner[count])
        zone.name='Coils'
        zone.render()

        count=count+1
        
#    MLab.draw()
#    MLab.show()

    return()
   

 









