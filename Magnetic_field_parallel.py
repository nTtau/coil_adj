#Runs full magnetic field code for any coil set 
#Vacuum field calculation -> doesnt currently include plasma contribution
#Final output is a csv file named B_tot.csv 
#Should also output mayavi magnetic field map -> Currently reworking this section so commented out for now  (field map call)

import parallel_loop as p_l
import parallel_filament as p_fil
import parallel_FieldVis as p_field
from grid_val_gen import grid_gen
import numpy as np
from pylab import *
import multiprocessing as mp
import logging
import csv


#INPUTS - Example case is a tokamak/torus

coil_data_path =  "Example_files/pf_example.csv"
TF_datafile_path =  "Example_files/TF_vals.csv"
bound_b_path = "Example_files/bounding_box.csv"
TF_coords_dir =  "Example_outputs/TF_gen_coord_output/new_TF"
TF_coil_num = 12 # number of TF coils 
PF_coil_num = 16 # number of PF coils
nproc_PF = 16 # number of processors to use for PF coil step 
nproc_TF = 12 # number of processors to use for TF coil step
geometry="Torus" # geometry tdye for reactor

#dx,dy,dz for sampling points 
#Would ideally add in a function to scale these based on the geometry to so same number of points in each direction 
dx = 0.5 
dy = 0.5
dz = 0.5  

start_t=time.perf_counter()



#Create list to store magnetic field arrays to
fields = []


start_t=time.perf_counter()



#Create list to store magnetic field arrays to
fields = []

if geometry =="Torus":


    #Set min/max values of x,y,z to sample and dx,dy,dz
    #Generate grid.csv file to be read by other codes when creating mesh grid 
    print("Generating sampling grid") 
    grid_gen(bound_b_path,dx,dy,dz)


    print("PF FIELD CALCULATING")
    PF_B=p_l.PF_FIELD(PF_coil_num,nproc_PF,coil_data_path,"grid_vals.csv")
    print(np.shape(PF_B))

    #Add PF field to array
    fields.append(PF_B)


    print("TF FIELD CALCULATING") 
    TF_B=p_fil.TF_FIELD(TF_coil_num,nproc_TF,TF_datafile_path,TF_coords_dir,"grid_vals.csv")


    #Add TF field to array
    fields.append(TF_B)
    #print(fields)


    #Calculate total field from arrays
    test=p_field.field_sum(fields)
    print(np.shape(test))
   


    #Pass field list to mayavi field plotting code 
   # print("Total field and mapping") 
   # p_field.field_map("Torus",coil_data_path,TF_coords_dir)

elif geometry == "stellarator":

    print("Generating sampling grid") 
    grid_gen(bound_b_path,dx,dy,dz)


    #Stellarator only has TF coils for now, use filament model approach 
    print("TOROIDAL FIELD COILS CALCULATING") 
    TF_B=p_fil.TF_FIELD(TF_coil_num,nproc_TF,TF_datafile_path,"new_TF","grid_vals.csv")


    #Add TF field to array
    fields.append(TF_B)
    #print(fields)

    #Calculate total field from arrays
    p_field.field_sum(fields)

    #Pass field list to mayavi field plotting code 
#    print("Field mapping") 
    #p_field.field_map("stellarator",coil_data_path,TF_coords_dir)

elif geometry == "cylinder":

    print("Generating sampling grid") 
    grid_gen(bound_b_path,dx,dy,dz)
 
    #Cylindrical geoms only have PF coils, all coils are circular so use loop code 
    print("PF FIELD CALCULATING")
    PF_B=p_l.PF_FIELD(PF_coil_num,nproc_PF,coil_data_path,"grid_vals.csv")


    #Add PF field to array
    fields.append(PF_B)

    #Calculate total field from arrays
    final_field=p_field.field_sum(fields)

elif geometry=="comsol": 

    print("Generating sampling grid") 
    grid_gen("bounding_box.csv",dx,dy,dz)
 
    #Cylindrical geoms only have PF coils, all coils are circular so use loop code 
    print("PF FIELD CALCULATING")
    PF_B=p_l.PF_FIELD(20,20,"comsol_comp/Bill_comp/comsol_coils.csv","grid_vals.csv")


    #Add PF field to array
    fields.append(PF_B)

    #Calculate total field from arrays
    final_field=p_field.field_sum(fields)
    
    



#Calculate total time taken to run
end_t=time.perf_counter()
tot_t = end_t-start_t#
print("TOTAL TIME TAKEN: ", tot_t)
