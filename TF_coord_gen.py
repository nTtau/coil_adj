####################################################################################
# Script to generate coords of TF coils as x,y,z coords                            #
# Allows for calculation of overall magnetic field using current filament approach #
####################################################################################

from CurrentFilament import CurrentFilament
import numpy as np
from numpy import linspace
from pylab import *
import csv



def RZ_to_XYZ(RZ_coords): 

    nrow,ncol=np.shape(RZ_coords)
    print(nrow,ncol)

    count = 0 
    while count < nrow: 
        print(count,RZ_coords[count,:])

        X_val = 0
        Y_val = 0
        Z_val = 0
        count=count+1
    
    return()

#Revolve TF coil around to generate others using the xyz coords 
#Write new coords to an array/file 
def revolve_TF(xyz):

    return() 




def rev(coords,points):

    coil_count = 1

    coil_num = 12
    angle_tot = 2 * pi

    # angle to rotate by
    angle_int = angle_tot / coil_num
    angle_orig = 0.0

    # print(angle_tot,angle_int)

    # Centre system around 0,0,0
    # already centred so no change

    while coil_count <= coil_num:
        filename = "new_TF/c_" + str(coil_count) + ".csv"
        TF_file = open(filename, "w")
        print("X,Y,Z", file=TF_file)
        count = 0
        #set angle for coil 
        angle = angle_orig+(angle_int*(coil_count-1))
        #revolve orig coords around by that angle 
        # new_coords
        while count <= points:

            x1 = (coords[count,0] * cos(angle)) + (coords[count,2] * sin(angle))  # + x_centre
            z1 = ((-coords[count,0]) * sin(angle)) + (coords[count,2] * cos(angle))
            y1 = coords[count,1]  # + y_centre

            print(x1, ",", y1, ",", z1, file=TF_file)
            count=count+1

        coil_count=coil_count+1



    return()



#NEED TO TURN THIS SECTION INTO FUNCTION
#MAKE INPUT COIL NUMBERS AND INPUT FILE WITH R Z COORDS FOR ONE COIL



#Read in R,Z from tokamak radial build 
 
TF_R_str = []
TF_Z_str = []
line_type=[]

tot_point=0 

RZ_file = open('Example_files/tf_coil_points.csv','r') 
file3 = csv.DictReader(RZ_file)

for col in file3:

    #Read in values
    TF_R_str.append(col['R'])
    TF_Z_str.append(col['Z'])
    line_type.append(col['Connection'])

    tot_point=tot_point+1
            
rz_coords = np.zeros((tot_point+1,3))

#Assign R,Z coords to array 
#Repeat first coord as last to close loop 

count=0
line_count =0

#check when first straight line is: 
s_lines= [index for (index, item) in enumerate(line_type) if item == "straight"]
print(s_lines)

inner= open("inner.csv", "w")
#print("X,Y,Z",file=inner)
outer= open("outer.csv", "w")
#print("X,Y,Z",file=outer)

while count <= tot_point-1:             

    rz_coords[count,0]=float(TF_R_str[count])
    rz_coords[count,1]=float(TF_Z_str[count])            
 
#    if line_count=0:


    rz_coords[count,2]=0.0
    count=count+1
    #First loop 

#Make new loop 
count = 0
while count <= tot_point-1:

    if count <= s_lines[0]: 

        if count ==0: 
            outer_ar = [[rz_coords[count,0], rz_coords[count,1], 0.0]]

        else: 
            outer_ar=np.append(outer_ar, [[rz_coords[count,0], rz_coords[count,1], 0.0]], axis=0)            
        print(rz_coords[count,0],",",rz_coords[count,1],",",rz_coords[count,2],file=outer)
        print(count,"OUTER")

    elif count > s_lines[0]:

        new_point = s_lines[0]+s_lines[1]-(count)+1
        print("NEW POINTS:",count,new_point)
        if count ==s_lines[0]+1: 
            inner_ar = [[rz_coords[new_point,0], rz_coords[new_point,1], 0.0]]

        else: 
#            print(count,new_point, [[rz_coords[new_point,0], rz_coords[new_point,1], 0.0]])
            inner_ar=np.append(inner_ar, [[rz_coords[new_point,0], rz_coords[new_point,1], 0.0]], axis=0)             
  
        print(rz_coords[count,0],",",rz_coords[count,1],",",rz_coords[count,2],file=inner)
        print(count,"INNER")


   
#Two loops go in opposite directions, need to swap them 


#    print(rz_coords[count,0],",",rz_coords[count,1],",",rz_coords[count,2],file=xyz)
      
#   print("RZ COORDS") 
 #   print(rz_coords[count,:])
    count=count+1
print(outer_ar)
print(inner_ar)

#Find middle of each array 
count =0 
mid= open("mid.csv", "w")

while count <= s_lines[0]:

    x_mid =(inner_ar[count,0] + outer_ar[count,0])/2
    y_mid =(inner_ar[count,1] + outer_ar[count,1])/2
    z_mid =(inner_ar[count,2] + outer_ar[count,2])/2

    if count ==0: 
        mid_line = [[x_mid,y_mid,z_mid]]

    else: 
        mid_line=np.append(mid_line, [[x_mid,y_mid,z_mid]], axis=0) 

    #Print to midline file 
    print(x_mid,",",y_mid,",",z_mid,",",file=mid)

    count=count+1

#Convert to RZ to XYZ 

#RZ_to_XYZ(rz_coords) 

print("MIDDLE LINE")
print(mid_line)
coil_num=12
#Midline made, revolve around by 360/n_coil degrees to get the next TF coil 



rev(mid_line,s_lines[0])
        

    











