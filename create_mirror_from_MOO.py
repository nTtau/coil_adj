import csv
import numpy as np
from sol_coil_gen import gen_so_coil_array
import json
import os
from typing import Dict, Any
# Load the CSV file


# Function to get user input and process the data
def save_json(settings: Dict[str, Any], output_dir: str, filename: str) -> None:
    """
    Save configuration to JSON file.
    :param settings:
    :param output_dir:
    :param filename:
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    print(f'Saving json file {file_path}')
    with open(file_path, 'w') as file:
        json.dump(settings, file, indent=4)


def process_data(data, selected_index):
    # Get user input
    selected_row = data[selected_index]

    # Extract the required values
    radial_build = [
        selected_row.get('Plasma', 'N/A'),
        selected_row.get('Vacuum', 'N/A'),
        selected_row.get('First_Wall', 'N/A'),
        selected_row.get('Blanket', 'N/A')
    ]

    # Extract other parameters
    parameters = {key: value for key, value in selected_row.items()
                  if key not in ['output', 'Plasma', 'Vacuum', 'First_Wall', 'Blanket']}

    return radial_build, parameters


file_path = 'InitialMOO.csv'
row_to_opt = 5
customer_name = f"MOO_{row_to_opt}"
# Read the CSV file into a list of dictionaries
data = []
with open(file_path, mode='r') as file:
    csv_reader = csv.DictReader(file)
    headers = csv_reader.fieldnames  # Get the headers
    for row in csv_reader:
        data.append(row)


# Process the data
radial_build, parameters = process_data(data, row_to_opt)

# Display the results
print("Radial Build:", radial_build)
print("Other Parameters:", parameters)

# extract current_density, coil_n, B_targ
radius = float(parameters['radius'])
sol_l = float(parameters['Reactor_Height'])
max_c_t = float(parameters['current_density'])
n_coils = int(parameters['coil_number'])
B_targ = 3.0

end_cell_height = 8
end_cell_coil_r = 2.6
end_cell_coil_z = [-54, -48, 48, 54]
end_cell_dr = 1.5
end_cell_dz = 3
expander_coil = False
nbi_cutter_height = 0

t_dr = 1.0 * (10**-3)
t_dz = 1.0 * (10**-3)


# Turn the strings in radial build to floats
radial_build = [float(i) for i in radial_build]
shielding_thickness = 0.2
radial_build.append(shielding_thickness)

coil_info = gen_so_coil_array(
    n_coils, B_targ, radius, sol_l, max_c_t, t_dr, t_dz)

# Take result of coil_info and save to csv with headers
# Save coil info to csv
# Before csv save , round each number to 4 dp
# Round each number to 4 decimal places
coil_info = np.round(coil_info, 4)
# Save coil info to CSV
with open('coil_info.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["nr", "nz", "I_per_turn", "r", "dr", "dz",
                    "Bc", "drdz", "z", "hoop_stress", "coil_vol", "coil_num"])
    writer.writerows(coil_info)

data = np.loadtxt('coil_info.csv', delimiter=',', skiprows=1)
r_vals = data[:, 3]
z_vals = data[:, 8]
dr_vals = data[:, 4]
dz_vals = data[:, 5]
# create dict settings
settings = {}
settings['customer_name'] = customer_name
settings['radial_build'] = radial_build
settings["component_names"] = ["Plasma", "Vacuum",
                               "First Wall", "Blanket", "Shielding"]
settings['reactor_height'] = sol_l
settings['end_cell_radius'] = float(parameters['End_Cell_Radius'])
settings['nbi_radius'] = float(parameters['nbi_radius'])
settings['shielding_thickness'] = shielding_thickness
settings['end_cell_shield'] = shielding_thickness
settings['solenoid_coil_r'] = r_vals.tolist()
settings['solenoid_coil_z'] = z_vals.tolist()
settings['solenoid_coil_dr'] = dr_vals.tolist()
settings['solenoid_coil_dz'] = dz_vals.tolist()
settings['end_cell_height'] = end_cell_height
settings['end_cell_coil_r'] = end_cell_coil_r
settings['end_cell_coil_z'] = end_cell_coil_z
settings['end_cell_coil_dr'] = end_cell_dr
settings['end_cell_coil_dz'] = end_cell_dz
settings['expander_coil'] = expander_coil
settings['current_density'] = max_c_t


save_json(settings, '.', 'settings.json')
