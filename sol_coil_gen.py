# Code to iterate through changing coil radii -> modify currrent until target B centre field hit, calculate hoop stress
# Output number of coils/spacing between coils, coil radius,coil current,Bcentre,coil dr/dz and hoop stress as csv -> maybe do average,min,max hoop stress?

# Use csv for NN training/M.O.O
# Train simple neural network? -> Then do M.O.O to optimise for min hoop stress and cost
# Could we also add some sort of field uniformity measure? then make that another to optimise
# Higher weighting to field uniformity -> higher weighting to physics considerations?
# Higher weighting to min hoop stress -> higher weighting to engineering considerations?
# Higher weighting to min cost -> higher weighting to cost considerations


# Would be valid for any solenoid set up? would have to assume evenly spaced coils
# Run each coil as a seperate process? for 16 coils use 16 threads?

# from CurrentFilament import CurrentFilament
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
from update_json import *

u0 = (np.pi)*4*(10**-7)  # mag const


# print("Number of processors available: ", mp.cpu_count())

# Creates a solenoid with the central axis in z direction
# Bc = target central field

# Central cell of a magnetic mirror acts like a solenoid
# Inputs are number of coils(n_coils), target field at centre(Bc), coil average radius (r), solenoid length(l), max allowed current per turn(max_c_turn)
# For now use max current load ratings at room temp -> add in adjustment for larger temps later, we will assume we can add sufficient cooling to maintain this temp
# Use AWG guide here:https://www.engineeringtoolbox.com/wire-gauges-d_419.html for copper coil
# Need to find same for YBCO/REBCO HTS

def gen_so_coil_array(n_coils, Bc, r, l, max_c_turn, turn_dr, turn_dz):

   # Initial guess for current per coil:
    # coil z spacing
    coil_z_s = l/n_coils
    I = c_e.sol_I_calc(n_coils, l, Bc)
    nturns = I/max_c_turn
#    print(nturns)
    nz_float = (np.sqrt(nturns))
    nz_int = int(nz_float)

    # Round number of z turns up if non integer
    if nz_float - nz_int > 0.000001:
        nz = nz_int + 1
    else:
        nz = nz_int

    nr = nz
    N_tot = nr*nz
    I_per_turn = I/N_tot

    dr = turn_dr*nr
    dz = turn_dz*nz

    coil_ar = np.zeros((n_coils, 12))

    for c_n in range(n_coils):
        # Set coil z spacing
        coil_z = -l/2 + (0.5*coil_z_s) + (c_n*coil_z_s)

        # Add values to the data array
        coil_ar[c_n, 0] = nr
        coil_ar[c_n, 1] = nz
        coil_ar[c_n, 2] = I_per_turn
        coil_ar[c_n, 3] = r
        coil_ar[c_n, 4] = dr
        coil_ar[c_n, 5] = dz
        coil_ar[c_n, 6] = 0.0
        coil_ar[c_n, 7] = 0.0
        coil_ar[c_n, 8] = coil_z
        coil_ar[c_n, 9] = 0.0
        coil_ar[c_n, 10] = 0.0
        coil_ar[c_n, 11] = 1.0

#    print("number of coils:",n_coils,"coil average radius:",r,"dr:",dr,"dz:",dz)
 #   print("Current needed per coil:",I)
  # Initial coil set up defined
    return (coil_ar)


# Calculates the total volumes of the coils given above

def pancake_coil_vol(coil_data):

    c_n = np.size(coil_data[:, 0])
    volumes = np.zeros((c_n, 1))
    for i in range(c_n):
        r_av = coil_data[i, 3]
        dr = coil_data[i, 4]
        dz = coil_data[i, 5]
        rmax = r_av + (dr/2)
        rmin = r_av - (dr/2)
        vout = np.pi*(rmax**2)*dz
        vin = np.pi*(rmin**2)*dz
#        print(rmax,rmin)
 #       print(vout,vin)

        volumes[i, 0] = vout-vin

    return (volumes)


# INPUTS
samp_points = 1000
sol_l = 20
B_targ = 3.0


# Use 400 AWG Cu for now
max_c_t = 415.0
# Ensure measurements in metres
t_dr = 13.34 * (10**-3)
t_dz = 13.34 * (10**-3)
radius = 7

# FOR HTS TAPE COILS
# Assuming critical current density od 1000A/mm2 for now ( will need updating to depend on field and temperature in future)
max_c_t = 200.0
t_dr = 1.0 * (10**-3)
t_dz = 1.0 * (10**-3)

# NOTE FOR DOM, COIL INFO IS THE ARRAY YOU NEED TO FEED BACK IN
coil_n = 20
coil_info = gen_so_coil_array(
    coil_n, B_targ, radius, sol_l, max_c_t, t_dr, t_dz)

# Take result of coil_info and save to csv with headers
# Save coil info to csv
# Before csv save , round each number to 4 dp
# Round each number to 4 decimal places
coil_info = np.round(coil_info, 4)

# Save coil info to CSV


for i in range(2, 21):
    coil_n = i
    coil_info = gen_so_coil_array(
        coil_n, B_targ, radius, sol_l, max_c_t, t_dr, t_dz)

    # Take result of coil_info and save to csv with headers
    # Save coil info to csv
    # Before csv save , round each number to 4 dp
    # Round each number to 4 decimal places
    coil_info = np.round(coil_info, 4)

    # Save coil info to CSV
    with open('coil_info.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["nr", "nz", "I_per_turn", "r", "dr", "dz",
                        "Bc", "drdz", "z", "hoop_stress", "coil_vol", "coil_num"])
        writer.writerows(coil_info)

    filename = f'coil_{i}'
    create_updated_json(
        'coil_info.csv', 'novatron.json', output_filename=filename, output_dir='../Novatron_cR'
    )
