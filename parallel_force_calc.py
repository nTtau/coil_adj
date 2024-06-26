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


"ADDED JUNE 2024 - USED TO CALCULATE HOOP STRESS AND FORCES" 



#Hoop stress estimate 
#Mainly applicable to solenoids, uses 1/2 central solenoid field as mag field estimate in the coil 

def hoop_est(Bmag,Itot,r_av,dr,dz):


    F_m = 0.5*Bmag*Itot
#    print(F_m)
    F = F_m * 2*np.pi*r_av
    T = F_m*r_av 
    hoop_s= T/(dz*dr) 

     
    
    return(F,hoop_s)

#F_e,h_s = hoop_est( 0.002247940713933032,1000.0,0.025,0.01,0.01)
#print ("FORCE ESTIMATE ON COIL:", F_e,"N")
#print ("HOOP STRESS ESTIMATE ON COIL:", h_s,"N/m^2 =",(h_s*(10**-6)),"N/mm^2 = ",(h_s*(10**-6)),"MPa" )

#Hoop stress on coil with cross section rectangular cross section dz,dr 

def hoop_s_calc(Fr,r_av,dz,dr): 

    
    hoop_s =Fr/(2*np.pi*dr*dz)
#    print("INPUTS TO HOOP S")
 #   print(Fr,r_av,dz,dr)
  #  print("hoop calc values:",Fr,r_av,dz,dr)
    return(hoop_s)


#Read circular coil file 

def coil_file_reader(coil_csv): 

   #read coil_csv in 

# Calculates the magnetic field from one coil using its properties

    #Coil properties from coil_data file 
    #READ COIL DATA FOR CORRECT COIL 
    
    #Open coil_data_file 
    #Read line corresponding to c_coil_num 

    coil_counter =0
    
    Nr=[]
    Nz=[]
    I_val=[]
    r=[]
    dr=[]
    coil_dz=[]
    x_c=[]
    y_c=[]
    z_c=[]
    x_n=[]
    y_n=[]
    z_n=[]
    
    filename = open(coil_csv, 'r')
    file = csv.DictReader(filename)

    for col in file:

        #count number of coils
        coil_counter=coil_counter+1
            

        #Read in values
#            print(coil_counter)
        Nr.append(int(col['R_turns']))
        Nz.append(int(col['Z_turns']))
        I_val.append(float(col['I (A)']))
        r.append(float(col['R_av']))
        dr.append(float(col['dr']))
        coil_dz.append(float(col['dz']))
        x_c.append(float(col['Coil_X']))
        y_c.append(float(col['Coil_Y']))
        z_c.append(float(col['Coil_Z']))
        x_n.append(float(col['Normal_x']))
        y_n.append(float(col['Normal_y']))
        z_n.append(float(col['Normal_z']))


  #  print(coil_counter)
    coil_tot=coil_counter
 #   print(coil_tot)

    #Combine into one large array 
    coil_data = np.zeros((coil_tot,12))
    coil_data[:,0] = Nr[:]
    coil_data[:,1] = Nz[:]
    coil_data[:,2] = I_val[:]
    coil_data[:,3] = r[:]
    coil_data[:,4] = dr[:]
    coil_data[:,5] = coil_dz[:]
    coil_data[:,6] = x_c[:]
    coil_data[:,7] = y_c[:]
    coil_data[:,8] = z_c[:]
    coil_data[:,9] = x_n[:]
    coil_data[:,10] = y_n[:]
    coil_data[:,11] = z_n[:]

    #coil_data=np.concatenate((coil_data,Nr,Nz,I_val,r,dr,coil_dz,x_c,y_c,z_c,x_n,y_n,z_n),axis=1)
#    print(np.shape(coil_data))
#    print("COIL VALUES READ")
#    print(coil_data[1])
    return(coil_data) 

#    return() 

#Gives the Force,radial force and hoop stress per circular coil given the others in the system
#Only works for circular coils
#fil_coil is a true or false statement -> used to determine whether to include non circular coils in calculation (such as TF coils) 
#c_n = circular coil number 
#data array is array of values for every circular coil 
#fil array = coil data for filament modelled coils, set with default value to make optional

def circ_coil_forces(B_back,c_n,data_array,nfil,fil_coil,fil_ar=[0,0,[0,0,0]]): 

    #Main script to run B calc for given coil 

    #used to write forces with coords, needs to append 

  #  print("CALCULATING FORCES USING FILAMENT MODEL")

    circ_coil_tot = np.size(data_array[:,0])
    #print(coil_tot)

 #   print("coil x,y,z:")
  #  print(data_array[c_n,6],data_array[c_n,7],data_array[c_n,8])

    num_f=nfil  #Number of points to sample force at on coil

    Force_ar=np.zeros((num_f,7))
    #print(np.shape(Force_ar))

    centre=MagCoil(array([0.0,0.0,0.0]), array([0,0,1.0]), R=1.0, I=0)

    R = data_array[c_n,3]
   # array_in = np.zeros((num_f+1,3))

    #Split coil values off into own array
    coil_ar=data_array[c_n,:]

    #Call pf sample coords function 
    array_in = pf_sample_coords(coil_ar,nfil)

 #   print("fil array:") 
 #   print(array_in)
#    print(array_in)

    N_val = data_array[c_n,0]*data_array[c_n,1] 
    cc=CurrentFilament(array_in,(N_val*data_array[c_n,2]))
 #   print("coil current:",N_val*data_array[c_n,2])

 #       c_n = 10
    for i in range(num_f):
  #  print("filament number:",i) 
    #Define points along current loop 2 
        angle = (np.pi*2*i)/num_f
        angle_2 = (np.pi*2*(i+1))/num_f
 #   print("angle 1:",angle)

        x = array_in[i,0] 
        y = array_in[i,1]
        z = array_in[i,2]

        x_next = array_in[i+1,0]  
        y_next = array_in[i+1,1]
        z_next = array_in[i+1,2]

            #Assumes coils alligned in z for now 

    #Calculate B from coil one at given coordinate
        smidge =array([x,y,z])
#    print("coordinate:",smidge)
  #  print("MEASURE FIELD AT POINT:",centre.r0+smidge)

            #Calculate magnetic field at sample point from all other coils 
        #Start with circular coils 
        
        Bnet = np.zeros((1,3))
        Bnet[0,0] = B_back[0]
        Bnet[0,1] = B_back[1]
        Bnet[0,2] = B_back[2]


        circ_coil = True
        if circ_coil ==True:

            #Same coil as testing field at 

            for coil_o in range(circ_coil_tot):
                if coil_o ==c_n:
                    #Don't calculate field 
                    #Bnet = Bnet
                    point=np.zeros((1,3))

  #                  print("Bnet before:",Bnet)
                    point[0,0]=x
                    point[0,1]=y
                    point[0,2]=z
                    B= cc.B(point)

                    Bnet=Bnet+B
 #                   print("Bnet after same coil:",Bnet)

                    point2=np.zeros((1,3))
                    point2[0,0]=0.0
                    point2[0,1]=0.0
                    point2[0,2]=z
                    Bc=cc.B(point2)
 #                   print("B at coil centre:",Bc)


                else: 
                    N_val = data_array[coil_o,0]*data_array[coil_o,1]       
                    c1=MagCoil(array([data_array[coil_o,6],data_array[coil_o,7],data_array[coil_o,8]]), array([data_array[coil_o,9],data_array[coil_o,10],data_array[coil_o,11]]), R=data_array[coil_o,3], I=N_val*(data_array[coil_o,2])) 
                 #   print("COIL TOTAL CURRENT:",N_val*(data_array[coil_o,2]))
                    B=c1.B(centre.r0+smidge)#+c2.B(c3.r0+smidge)#
                    Bnet = Bnet + B
                 #   print("Bnet final:",Bnet)


        #Add in magnetic field from filament coils 

 #       print("fil_coil value:",fil_coil)
       # print("TF array:",fil_ar)
        if fil_coil ==True:
            print("adding TF coils")
         
            fil_coil_tot =np.size(fil_ar[:,0])
 #           print("number of TF :", fil_coil_tot)
            for coil_o in range(fil_coil_tot):


                c1=CurrentFilament(fil_ar[coil_o,2],(fil_ar[coil_o,0]*fil_ar[coil_o,1]))
                B=c1.B(point)#+c2.B(c3.r0+smidge)#
                Bnet = Bnet + B
#                print("B from TF coil:",B)
   
#                    print(np.shape(Bnet)) 
                    #print("added coil: ",coil_o,"Mag field:",(Bnet))
 #   print("FIELD AT POINT:",Bnet)

            #Check with estimated B field
  #          Bnet[0,0]=0.0
 #           Bnet[0,1]=0.0
#            Bnet[0,2]=0.5*2.7100
          
#            print(np.shape(Bnet))
  #      print("COIL TOTAL CURRENT:",N_val*(data_array[coil_o,2]))
   #     print("Net field on coil:", Bnet)
  #      print("at point:", x,y,z)
        dl = array([x-x_next,y-y_next,z-z_next])
        #Calculate force on coil filament
        F = (data_array[c_n,0]*data_array[c_n,1]*data_array[c_n,2])*(np.cross(Bnet,dl))
#            print("F per element =", F)
        F_mag = np.sqrt((F[0,0])**2+(F[0,1])**2 + (F[0,2])**2)
#            print(F_mag)
 #       Fr = (np.cos(angle)*F[0,0]) + ( np.sin(angle)*F[0,1])
    #    print("dot product")
       # Fc = (array[F[0,0],F[0,1],F[0,2]])
        #print("Fc:",Fc)
   #     print("F,dl:",F,dl)
        Fc = F[0,:]
        #vector to sample point
        p = [x-data_array[c_n,6],y-data_array[c_n,7],z-data_array[c_n,8]]
        p_l = np.sqrt((p[0]**2) + (p[1]**2) + (p[2]**2))
#        print("Fc")
 #       print(Fc,p,np.dot(Fc,p))
  #      print("dot prod:,",F[0,0]*p[0] + F[0,1]*p[1])
        Fr = (np.dot(Fc,p))/(p_l)


        #Set Force_array values
        Force_ar[i,0] =x
        Force_ar[i,1] =y
        Force_ar[i,2] =z
        Force_ar[i,3] =F[0,0]
        Force_ar[i,4] =F[0,1]
        Force_ar[i,5] =F[0,2]
        Force_ar[i,6] =Fr


        #Calcuate total forces on the coil 
        if i ==0: 
            Fr_sum = Fr 
            dl_sum = np.sqrt(((dl[0])**2) + ((dl[1])**2) + ((dl[2])**2))
        else: 
            Fr_sum = Fr_sum + Fr
            dl_sum = dl_sum + (np.sqrt(((dl[0])**2) + ((dl[1])**2) + ((dl[2])**2)))

#            print("dl total length:",dl_mag)
 #           print("Fr for element:",i,"=",Fr)

            #Add to total force on coil c_n
        if i ==0: 
            Force_tot = F 
        else:
            Force_tot = Force_tot + F 

 #   print("Coil:", c_n, "Tot force:",Force_tot,"Total radial force:",Fr_sum)
#    print("total length:",dl_sum, "should be:", (2*np.pi*0.13))
    Fr_sum = Fr_sum *(dl_sum/(2*np.pi*(data_array[c_n,3])))
    h_s = hoop_s_calc(Fr_sum,data_array[c_n,3],data_array[c_n,5],data_array[c_n,4])
  #  print("Hoop stress in coil:",c_n," =", h_s, "N/m^2, = ",h_s/(10**6),"N/mm^2 = MPa")
    
    #fplot.close()
    return(Force_tot,Fr_sum,h_s,c_n,Force_ar) 

def fil_coil_forces(B_back,c_n,fil_ar,circ_coil=True,data_array=[0,0,[0,0,0]]): 

    fil_coil=True
    #Main script to run B calc for given coil 

    #used to write forces with coords, needs to append 

  #  print("CALCULATING FORCES USING FILAMENT MODEL")

    fil_coil_tot =np.size(fil_ar[:,0])
    #print(coil_tot)

 #   print("coil x,y,z:")
  #  print(data_array[c_n,6],data_array[c_n,7],data_array[c_n,8])

      #Number of points to sample force at on coil
    

    #print(np.shape(Force_ar))

    centre=MagCoil(array([0.0,0.0,0.0]), array([0,0,1.0]), R=1.0, I=0)

    cc=CurrentFilament(fil_ar[c_n,2],(fil_ar[c_n,0]*fil_ar[c_n,1]))
    array_in=fil_ar[c_n,2]
    num_f=np.size(array_in[:,0])
    Force_ar=np.zeros((num_f,7))

 #       c_n = 10
    for i in range(num_f-1):

        x = array_in[i,0] 
        y = array_in[i,1]
        z = array_in[i,2]

        x_next = array_in[i+1,0]  
        y_next = array_in[i+1,1]
        z_next = array_in[i+1,2]

        point=np.zeros((1,3))

        point[0,0]=x
        point[0,1]=y
        point[0,2]=z

            #Assumes coils alligned in z for now 

    #Calculate B from coil one at given coordinate
        smidge =array([x,y,z])
#    print("coordinate:",smidge)
  #  print("MEASURE FIELD AT POINT:",centre.r0+smidge)

            #Calculate magnetic field at sample point from all other coils 
        #Start with circular coils 
        #Set initial B net to the background field 
        
        Bnet = np.zeros((1,3))
        Bnet[0,0] = B_back[0]
        Bnet[0,1] = B_back[1]
        Bnet[0,2] = B_back[2]

        if circ_coil ==True: 
            circ_coil_tot = np.size(data_array[:,0])

            for coil_o in range(circ_coil_tot):

                N_val = data_array[coil_o,0]*data_array[coil_o,1]       
                c1=MagCoil(array([data_array[coil_o,6],data_array[coil_o,7],data_array[coil_o,8]]), array([data_array[coil_o,9],data_array[coil_o,10],data_array[coil_o,11]]), R=data_array[coil_o,3], I=N_val*(data_array[coil_o,2])) 
                B=c1.B(centre.r0+smidge)#+c2.B(c3.r0+smidge)#
                Bnet = Bnet + B

        #Add in magnetic field from filament coils 

        if fil_coil ==True:
#            print("adding TF coils")
         
            fil_coil_tot =np.size(fil_ar[:,0])
#            print("number of TF :", fil_coil_tot)
            for coil_o in range(fil_coil_tot):


                c1=CurrentFilament(fil_ar[coil_o,2],(fil_ar[coil_o,0]*fil_ar[coil_o,1]))

                B=c1.B(point)#+c2.B(c3.r0+smidge)#
                Bnet = Bnet + B
#                print("B from TF coil:",B)

        elif fil_coil ==False: 
#            print("NOT INCLUDING NON CIRCULAR COILS IN CALCULATION") 
            Bnet=Bnet     
#                    print(np.shape(Bnet)) 
                    #print("added coil: ",coil_o,"Mag field:",(Bnet))
 #   print("FIELD AT POINT:",Bnet)


          
#            print(np.shape(Bnet))
        dl = array([x-x_next,y-y_next,z-z_next])
        #Calculate force on coil filament
        F = (data_array[c_n,0]*data_array[c_n,1]*data_array[c_n,2])*(np.cross(Bnet,dl))
#            print("F per element =", F)
        F_mag = np.sqrt((F[0,0])**2+(F[0,1])**2 + (F[0,2])**2)
#            print(F_mag)
 #       Fr = (np.cos(angle)*F[0,0]) + ( np.sin(angle)*F[0,1])
    #    print("dot product")
       # Fc = (array[F[0,0],F[0,1],F[0,2]])
        #print("Fc:",Fc)
   #     print("F,dl:",F,dl)
        Fc = F[0,:]
        #vector to sample point
        p = [x-data_array[c_n,6],y-data_array[c_n,7],z-data_array[c_n,8]]
        p_l = np.sqrt((p[0]**2) + (p[1]**2) + (p[2]**2))
#        print("Fc")
 #       print(Fc,p,np.dot(Fc,p))
  #      print("dot prod:,",F[0,0]*p[0] + F[0,1]*p[1])
        Fr = (np.dot(Fc,p))/(p_l)


        #Set Force_array values
        Force_ar[i,0] =x
        Force_ar[i,1] =y
        Force_ar[i,2] =z
        Force_ar[i,3] =F[0,0]
        Force_ar[i,4] =F[0,1]
        Force_ar[i,5] =F[0,2]
        Force_ar[i,6] =Fr


        #Calcuate total forces on the coil 
        if i ==0: 
            Fr_sum = Fr 
            dl_sum = np.sqrt(((dl[0])**2) + ((dl[1])**2) + ((dl[2])**2))
        else: 
            Fr_sum = Fr_sum + Fr
            dl_sum = dl_sum + (np.sqrt(((dl[0])**2) + ((dl[1])**2) + ((dl[2])**2)))

#            print("dl total length:",dl_mag)
 #           print("Fr for element:",i,"=",Fr)

            #Add to total force on coil c_n
        if i ==0: 
            Force_tot = F 
        else:
            Force_tot = Force_tot + F 

 #   print("Coil:", c_n, "Tot force:",Force_tot,"Total radial force:",Fr_sum)
#    print("total length:",dl_sum, "should be:", (2*np.pi*0.13))
    Fr_sum = Fr_sum *(dl_sum/(2*np.pi*(data_array[c_n,3])))
    h_s = hoop_s_calc(Fr_sum,data_array[c_n,3],data_array[c_n,5],data_array[c_n,4])
#    print("Hoop stress in coil:",c_n," =", h_s, "N/m^2, = ",h_s/(10**6),"N/mm^2 = MPa")
    
    #fplot.close()
    return(Force_tot,Fr_sum,h_s,c_n,Force_ar) 


#Calculate the force on a loop from its own field 
def F_r_from_single_loop(R,I,nfil):

    #Define array in 
    array_in = np.zeros((nfil,3))
 #   print(np.shape(array_in))

    for i in range(nfil):
        angle = (np.pi*2*i)/nfil
        x = (R*cos(angle)) 
        y = R*sin(angle) 
        z = 0.0

        array_in[i,0] = x
        array_in[i,1] = y
        array_in[i,2] = z

    c1=CurrentFilament(array_in,I)
          
    for j in range(nfil):
 
  #  print("filament number:",i) 
    #Define points along current loop 2 
        angle = (np.pi*2*j)/nfil
        angle_2 = (np.pi*2*(j+1))/nfil
 #   print("angle 1:",angle)

            #Assumes coils alligned in z for now 
        x = (R*cos(angle)) 
        y = R*sin(angle) 
        z = 0.0

        x_next = R*cos(angle_2) 
        y_next = R*sin(angle_2)
        z_next = 0.0

        point=np.zeros((1,3))
        point[0,0]=x
        point[0,1]=y
        point[0,2]=z

 #       point_coords[0,0]=centre[0,0]+x_current
  #      point_coords[0,1]=centre[0,1]+y_current
   #     point_coords[0,2]=centre[0,2]+z_current
                  
    #    Bnet=c1.B(point_coords)

        #Calculate force on each filament due to rest of coil
        Bnet= c1.B(point)
        
        dl = array([x-x_next,y-y_next,z-z_next])
        

        F = (I)*(np.cross(Bnet,dl))
#            print("F per element =", F)
        F_mag = np.sqrt((F[0,0])**2+(F[0,1])**2 + (F[0,2])**2)
#            print(F_mag)
 #       Fr = (np.cos(angle)*F[0,0]) + ( np.sin(angle)*F[0,1])
 #       print("dot product")
  #      print(F,dl,numpy.dot(F,dl))
 #       

 #       print("Fil:",j,"Fr per fil:", Fr)

        if j ==0: 
            Fr_sum = Fr 
            dl_sum = np.sqrt(((dl[0])**2) + ((dl[1])**2) + ((dl[2])**2))
        else: 
            Fr_sum = Fr_sum + Fr
            dl_sum = dl_sum + (np.sqrt(((dl[0])**2) + ((dl[1])**2) + ((dl[2])**2)))

  #      print("total length:",dl_sum)

           
 
    h_s = hoop_s_calc(Fr_sum,R,0.017,1.5)


    
    return(Fr_sum,h_s) 

def all_coils(proc_num,nfil,B_back,use_circ_c=False,data_array=[0,0,0],use_fil_c=False,fil_ar_in=[0,0,[0,0,0]]):

    #Count total coil number 
    circ_coil_max = np.size(data_array[:,0])
    print("total coils: ", circ_coil_max)
    fileout = open("CoilForces.csv", 'w')
    print("coil num,Fx,Fy,Fz,FMag,Fr,hoop_s(MPa)",file=fileout)

    force_plot = open("force_plot.csv", 'w')
    print("x,y,z,Fx,Fy,Fz,Fr,hoop_s",file=force_plot)


#ADD circular coils 

#SPLIT COIL COORD ARRAY INTO ONE "CHUNK" PER COIL

    if use_circ_c==True:
        print("ADDING CIRCULAR COILS")

#Make pool
        pool = mp.Pool(proc_num)

#Compute B field from each coil
        out = [ pool.apply_async(circ_coil_forces, (B_back,c_n,data_array,nfil,use_fil_c,fil_ar_in)) for c_n in range(circ_coil_max)]

#Close pool and join results
        pool.close()
        pool.join()

        for i in range(circ_coil_max):
 #       print("VALS FOR COIL:",i)
 #       print("NET FORCE:",(out[i].get())[0])
 #       print("RADIAL FORCE:",(out[i].get())[1])
  #      print("HOOP STRESS:",(out[i].get())[2]*(10**-6),"MPa")
            np.shape(out)
            if i ==0:
                hs_sum = ((out[i].get())[2])
                max_hs=  ((out[i].get())[2])
                Fr_tot = (out[i].get())[1]


            else: 
                hs_sum = ((out[i].get())[2]) + hs_sum
                Fr_tot = (out[i].get())[1] + Fr_tot
                if ((out[i].get())[2]) > max_hs: 
                    max_hs=  ((out[i].get())[2])

        #Force array with coords
            Forces=(out[i].get())[4]
            nfil = np.size(Forces[:,0])
    #        print("nfil:",nfil)
            hs=((out[i].get())[2])
            Fr=(out[i].get())[1]
            for j in range(nfil):
                print(Forces[j,0],Forces[j,1],Forces[j,2],Forces[j,3],Forces[j,4],Forces[j,5],Forces[j,6],hs,sep=",",file=force_plot)
  #      print("FORCE ARRAY:")
  #      print(Forces)

            FMag= sqrt((((out[i].get())[0][0,0])**2) + (((out[i].get())[0][0,1])**2)+(((out[i].get())[0][0,2])**2))

            print(((out[i].get())[3]),(out[i].get())[0][0,0],(out[i].get())[0][0,1],(out[i].get())[0][0,2],FMag,Fr,hs*(10**-6),sep=",",file=fileout)
#            print(((out[i].get())[3]),(out[i].get())[0][0,0],(out[i].get())[0][0,1],(out[i].get())[0][0,2],FMag,sep=",")

#Add in TF coil values 

    if use_fil_c == True:
        print("ADDING NON-CIRCULAR COILS")
#Make pool
        fil_coil_max = np.size(fil_ar_in[:,0])
        pool = mp.Pool(proc_num)

#Compute B field from each coil
        fil_out = [ pool.apply_async(fil_coil_forces, (B_back,c_n,fil_ar_in,use_circ_c,data_array)) for c_n in range(fil_coil_max)]

#Close pool and join results
        pool.close()
        pool.join()

        for i in range(fil_coil_max):
 #       print("VALS FOR COIL:",i)
 #       print("NET FORCE:",(out[i].get())[0])
 #       print("RADIAL FORCE:",(out[i].get())[1])
  #      print("HOOP STRESS:",(out[i].get())[2]*(10**-6),"MPa")
            np.shape(fil_out)
            if i ==0 and use_circ_c ==False:
                hs_sum = ((fil_out[i].get())[2])
                max_hs=  ((fil_out[i].get())[2])
                Fr_tot = (fil_out[i].get())[1]


            else: 
                hs_sum = ((fil_out[i].get())[2]) + hs_sum
                Fr_tot = (fil_out[i].get())[1] + Fr_tot
                if ((fil_out[i].get())[2]) > max_hs: 
                    max_hs=  ((fil_out[i].get())[2])

        #Force array with coords
            Forces=(fil_out[i].get())[4]
            nfil = np.size(Forces[:,0])
    #        print("nfil:",nfil)
            hs=((fil_out[i].get())[2])
            for j in range(nfil):

                print(Forces[j,0],Forces[j,1],Forces[j,2],Forces[j,3],Forces[j,4],Forces[j,5],Forces[j,6],hs,sep=",",file=force_plot)


            FMag= sqrt((((fil_out[i].get())[0][0,0])**2) + (((fil_out[i].get())[0][0,1])**2)+(((fil_out[i].get())[0][0,2])**2))


            print(((fil_out[i].get())[3]),(fil_out[i].get())[0][0,0],(fil_out[i].get())[0][0,1],(fil_out[i].get())[0][0,2],FMag,sep=",",file=fileout)
 #           print(((fil_out[i].get())[3]),(fil_out[i].get())[0][0,0],(fil_out[i].get())[0][0,1],(fil_out[i].get())[0][0,2],FMag,sep=",")
  #      print("FORCE ARRAY:")
  #      print(Forces)
 
#    print("TOTAL RADIAL FORCE:", Fr_tot)
#    print("MAX HOOP STRESS:",max_hs*(10**-6),"Mpa")            


    return(hs_sum)



#Calculate sampling coordinates for forces on circular coils
def pf_sample_coords(coil_ar,nfil): 

    coil_coord = np.zeros((nfil+1,3))
    R =coil_ar[3] 
    
    #set coil normal vector 
    vn = [coil_ar[9],coil_ar[10],coil_ar[11]]
    v1,v2 = perp_vector(vn)
    #determine coil orientation
    # then set sampling coordinates 

    #Check coil orientation 
    
    for i in range(nfil):
        angle = (np.pi*2*i)/nfil
#        print("Perpendicular vectors:",v1,v2)
        change_vector = [] 
        change_vector.append((R*cos(angle)*v1[0]) + (R*sin(angle)*v2[0]))
        change_vector.append((R*cos(angle)*v1[1]) + (R*sin(angle)*v2[1]))
        change_vector.append((R*cos(angle)*v1[2]) + (R*sin(angle)*v2[2]))
#        print(change_vector)
        x = coil_ar[6] + change_vector[0]
        y = coil_ar[7] + change_vector[1] 
        z = coil_ar[8] + change_vector[2]

        coil_coord[i,0] = x
        coil_coord[i,1] = y
        coil_coord[i,2] = z

    coil_coord[nfil,:] = coil_coord[0,:]


    return(coil_coord)

#Calculates two perpendicular vectors to the coil normal and normalises these
#Used to define vectors for generating circle perpendicular to the normal 
def perp_vector(vector_in): 

    i1 = vector_in[0]
    j1 = vector_in[1]
    k1 = vector_in[2]

    #Count how many zero values in vector 
    #if two, vector follows a cartesian axis 
    #make perpendicular vectors the other two axes 
    #if three zeros, invalid vector exit 
    count_zeros = 0
 
    if i1 ==0: 
      count_zeros = count_zeros+1 
    if j1 ==0: 
      count_zeros = count_zeros+1 
    if k1 ==0: 
      count_zeros = count_zeros+1  

    if count_zeros ==3: 
       print("Invalid normal vector,cannot be (0,0,0)") 

    elif count_zeros ==2: 
  
#       print("Vector alligns with a cartesian axis -> normal vectors are other two axes") 
       if i1 != 0: 
 #          print("Normal in x direction") 
           v1 = [0,1,0] 
           v2 = [0,0,1]
       elif j1 != 0: 
 #          print("Normal in y direction") 
           v1 = [1,0,0] 
           v2 = [0,0,1]
       elif k1 != 0: 
  #         print("Normal in z direction") 
           v1 = [1,0,0] 
           v2 = [0,1,0] 

    elif count_zeros ==1:

       #Use a.b = 0 for perpendicular vectors 
       if i1 == 0: 
           v1 = [0,-k1,j1] 
           
       elif j1 == 0: 
           v1 = [-k1,0,i1] 
       elif k1 == 0: 
           v1 = [j1,-i1] 

       #Calculate second vector 
       #Cross product gives a perpendicular vector to the other two 
       v2 = np.cross(vector_in,v1) 

       

    #No zero values, can just set x to 0 
    elif count_zeros ==0: 
        v1 = [0,-k1,j1]
        v2 = np.cross(vector_in,v1)

       #Normalise vectors
        v1_l = np.sqrt((v1[0]**2) +  (v1[1]**2) + (v1[2]**2))
        v1 = v1/v1_l
        v2_l = np.sqrt((v2[0]**2) +  (v2[1]**2) + (v2[2]**2))
        v2 = v2/v2_l

    print(vector_in,v1,v2)
    
    return(v1,v2)
 
#Gives the Force,radial force and hoop stress per circular coil given the others in the system
#Making work for TF and PF coils
#Make coordinate array for sampling points on the coil 
#Calculate total magnetic field from all coils(TF and PF) at these sampling points
def Net_per_coil(B_back,c_n,sample_coords,PF_data_array,TF_data_array): 

    #Main script to run B calc for given coil 

    
 #   print("CALCULATING FORCES USING FILAMENT MODEL")

    coil_tot = np.size(data_array[:,0])
    #print(coil_tot)

    Force_ar=np.zeros((coil_tot,3))

    num_f=100  #Number of points to sample force at on coil

    centre=MagCoil(array([0.0,0.0,0.0]), array([0,0,1.0]), R=1.0, I=0)

    R = data_array[c_n,3]
    array_in = np.zeros((num_f,3))

#    print("coil x,y,z:")
#    print(data_array[c_n,6],data_array[c_n,7],data_array[c_n,8])
      #  print(np.shape(array_in))

    for i in range(num_f):

        angle = (np.pi*2*i)/num_f
        angle_2 = (np.pi*2*(i+1))/num_f

        x = data_array[c_n,6]+(R*cos(angle)) 
        y = data_array[c_n,7]+R*sin(angle) 
        z = data_array[c_n,8]

        x_next = data_array[c_n,6]+R*cos(angle_2) 
        y_next = data_array[c_n,7]+R*sin(angle_2)
        z_next = data_array[c_n,8]

        array_in[i,0] = x
        array_in[i,1] = y
        array_in[i,2] = z

    cc=CurrentFilament(array_in,data_array[c_n,2])

 #       c_n = 10
    for i in range(num_f):
  #  print("filament number:",i) 
    #Define points along current loop 2 
        angle = (np.pi*2*i)/num_f
        angle_2 = (np.pi*2*(i+1))/num_f
 #   print("angle 1:",angle)
    

            #Assumes coils alligned in z for now 
        x = data_array[c_n,6]+(R*cos(angle)) 
        y = data_array[c_n,7]+R*sin(angle) 
        z = data_array[c_n,8]

        x_next = data_array[c_n,6]+R*cos(angle_2) 
        y_next = data_array[c_n,7]+R*sin(angle_2)
        z_next = data_array[c_n,8]

    #Calculate B from coil one at given coordinate
        smidge =array([x,y,z])
#    print("coordinate:",smidge)
  #  print("MEASURE FIELD AT POINT:",centre.r0+smidge)

            #Calculate magnetic field at sample point from all other coils 
        Bnet = np.zeros((1,3))
        Bnet[0,0] = B_back[0]
        Bnet[0,1] = B_back[1]
        Bnet[0,2] = B_back[2]
        for coil_o in range(coil_tot):
            if coil_o ==c_n:
                    #Don't calculate field 
                    #Bnet = Bnet
                point=np.zeros((1,3))

                point[0,0]=x
                point[0,1]=y
                point[0,2]=z
                B= cc.B(point)

                Bnet=Bnet+B


            else: 
                N_val = data_array[coil_o,0]*data_array[coil_o,1]       
                c1=MagCoil(array([data_array[coil_o,6],data_array[coil_o,7],data_array[coil_o,8]]), array([data_array[coil_o,9],data_array[coil_o,10],data_array[coil_o,11]]), R=data_array[coil_o,3], I=N_val*(data_array[coil_o,2])) 
                B=c1.B(centre.r0+smidge)#+c2.B(c3.r0+smidge)#
                Bnet = Bnet + B
#                    print(np.shape(Bnet)) 
                    #print("added coil: ",coil_o,"Mag field:",(Bnet))
 #   print("FIELD AT POINT:",Bnet)

            #Check with estimated B field
  #          Bnet[0,0]=0.0
 #           Bnet[0,1]=0.0
#            Bnet[0,2]=0.5*2.7100
          
#            print(np.shape(Bnet))
        dl = array([x-x_next,y-y_next,z-z_next])

        #Calculate force on coil filament
        F = (data_array[c_n,0]*data_array[c_n,1]*data_array[c_n,2])*(np.cross(Bnet,dl))
#            print("F per element =", F)
        F_mag = np.sqrt((F[0,0])**2+(F[0,1])**2 + (F[0,2])**2)
#            print(F_mag)
        Fr = (np.cos(angle)*F[0,0]) + ( np.sin(angle)*F[0,1])


        #Calcuate total forces on the coil 
        if i ==0: 
            Fr_sum = Fr 
        else: 
            Fr_sum = Fr_sum + Fr
        dl_mag = np.sqrt((dl[0])**2 +(dl[1])**2 +(dl[2])**2)
#            print("dl total length:",dl_mag)
 #           print("Fr for element:",i,"=",Fr)

            #Add to total force on coil c_n
        if i ==0: 
            Force_tot = F 
        else:
            Force_tot = Force_tot + F 

#    print("Coil:", c_n, "Tot force:",Force_tot,"Total radial force:",Fr_sum)
    h_s = hoop_s_calc(Fr_sum,data_array[c_n,3],data_array[c_n,5],data_array[c_n,4])
 #   print("Hoop stress in coil:",c_n," =", h_s, "N/m^2, = ",h_s/(10**6),"N/mm^2 = MPa")
    
    return(Force_tot,Fr_sum,h_s) 

#Create data array for all coils mapped with filament method and corresponding current etc 

def fil_file_reader(vals_file,coord_dir):

    filename = open(vals_file, 'r')
    file = csv.DictReader(filename)

    N=[] 
    Nr=[]
    Nz=[]
    I_val=[]

    coil_counter = 0 

    for col in file:

        #count number of coils
        coil_counter=coil_counter+1
            

        #Read in values
        #coil_num,N,I (A)
#            print(coil_counter)
        N.append((int(col['R_turns']))*(int(col['Z_turns'])))
        I_val.append(float(col['I (A)']))

  #  print(coil_counter)
    coil_tot=coil_counter


    for c_n in range(coil_tot): 

        fil_count=0
        #read filament coordinates in 
        coil_filename = coord_dir +"/c_" + str(c_n+1) + ".csv" 
 #       print("coil file:",coil_filename)
        coil_file= open(coil_filename, 'r')
        x = []
        y = []
        z = []
        file2 = csv.DictReader(coil_file)
        

        for col in file2:
            x.append(col['X'])
            y.append(col['Y'])
            z.append(col['Z'])
            fil_count = fil_count+1

        coords = np.zeros((fil_count+1,3))

        #Set coordinate array 
        for i in range(fil_count): 
            coords[i,0] = x[i]
            coords[i,1] = y[i]
            coords[i,2] = z[i]

        coords[fil_count,0] = x[0]
        coords[fil_count,1] = y[0]
        coords[fil_count,2] = z[0]        


        coil_vals=np.array([N[c_n],I_val[c_n],coords],dtype = object)
 #       print("Coil vals shape:", np.shape(coil_vals))
        if c_n ==0: 
            data_array = coil_vals
     #       print("data array shape:", np.shape(data_array))
        else: 
            data_array=np.vstack((data_array,coil_vals))
 #       print(data_array)

    return(data_array)



#a,F_s=Force_coils("comsol_comp/Bill_comp/solenoid.csv",1)
#print("Solnoid total radial force:",F_s)

#Calculate hoop stress from radial force 
#h_s=hoop_s_calc(F_s,0.13,0.017,1.5)
#print("Hoop stress =", h_s, "N/m^2, = ",h_s/(10**6),"N/mm^2 = MPa")
#F_r_from_single_loop(0.13,36759.36611753151,100)

#a=Force_coils("comsol_comp/comsol_coils.csv",1)

#dat=coil_file_reader("comsol_comp/comsol_coils.csv")

#dat=coil_file_reader("/home/ssharpe/GITHUB/SBRI-Year-2/mag_field_calc/Example_files/pf_yuhu.csv")
#dat=coil_file_reader("comsol_comp/Bill_comp/solenoid.csv")
#print(dat[0])

#sum_hs = 0.0 
#coil_tot = np.size(dat[:,0])
#for coil_n in range(coil_tot):
 #   F,Fr,hs=circ_coil_forces(coil_n,dat)
#    sum_hs = sum_hs+hs

#hs_av=(sum_hs)/(coil_tot)
#print("Hoop stress =", hs_av, "N/m^2, = ",hs_av/(10**6),"N/mm^2 = MPa")

#Fr_single,hs_single=F_r_from_single_loop(0.13,3234824.218342772,200)
#print("Hoop stress =", hs_single, "N/m^2, = ",hs_single/(10**6),"N/mm^2 = MPa")

#hs_tot=all_coils(dat,30)
#coil_tot = np.size(dat[:,0])
#hs_av=(hs_tot)/(coil_tot)


#Check convergence
#Length = 1.5 
#dr = 0.017
#R = 0.13 +(0.5*dr)
#Bc = 2.713
#t = 0.017

#hs_file = open("hs_converge.csv","w")
#print("N turns,average hoop stress(MPa)")
#
#Check convergence for the solenoid

#for i in range(100,110,10):

 #   print("Using ", i,"coils")

#    c_e.sol_gen(i,Length,R,Bc,t,(Length/i))
 #   dat=coil_file_reader("comsol_comp/Bill_comp/solenoid.csv")
  #  hs_tot=all_coils(dat,30)
   # hs_av=(hs_tot*(10**-6))/(i)
#    print(i,hs_av,sep=",",file=hs_file)
#Write hoop stress to a file: 

#print("AVERAGE HOOP STRESS =", hs_av, "N/m^2, = ",hs_av/(10**6),"N/mm^2 = MPa")

#circ_dat=coil_file_reader("/home/ssharpe/GITHUB/mag_field_calc/Example_files/pf_example.csv")
#fil_dat=fil_file_reader("/home/ssharpe/GITHUB/mag_field_calc/Example_files/TF_vals.csv","/home/ssharpe/GITHUB/mag_field_calc/new_TF")

#call code to read in filament coil values 
#hs_tot=all_coils(16,100,True,circ_dat,True,fil_dat)
#coil_num = 28
#Background magnetic field (in T) - only valid for uniform background fields 
#B_back = [0,0,0] 
#i = 100
#Length = 92.0
#c_e.sol_gen(i,92.0,4.0,3.0,1.0,(Length/i))
#circ_dat=coil_file_reader("/home/ssharpe/REALTA/12_06/pf_coils.csv")
#circ_dat=coil_file_reader("comsol_comp/test.csv")
#hs_tot=all_coils(coil_num,200,B_back,True,circ_dat,False)
#print("total hoop stress:",hs_tot*(10**-6),"MPa")
#print("Average hoop stress:", hs_tot*(10**-6)/coil_num,"MPa")











