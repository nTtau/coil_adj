import numpy as np
import json
import os
from typing import Dict, Any


def load_json(filename: str) -> Dict[str, Any]:
    """
    Load json file. Intended to be used for configuration and parameters.
    :param filename:
    :return: Loaded JSON
    """
    print(f'Loading json file {filename}')
    with open(filename, 'r') as file:
        return json.load(file)


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


def create_updated_json(
        updated_coil_shielding: float,
        coil_path: str,
        settings_path: str,
        output_filename: str = 'settings',
        output_dir: str = 'output') -> None:

    # Load the CSV file
    data = np.loadtxt(coil_path, delimiter=',', skiprows=1)
    output_filename_json = f'{output_filename}.json'
    # Load json
    settings = load_json(settings_path)

    # Extract the 'r' column (4th column, index 3)
    r_vals = data[:, 3]
    z_vals = data[:, 8]
    dr_vals = data[:, 4]
    dz_vals = data[:, 5]
    radial_build = settings['radial_build']
    radial_build.append(updated_coil_shielding)

    settings['solenoid_coil_r'] = r_vals.tolist()
    settings['solenoid_coil_z'] = z_vals.tolist()
    settings['solenoid_coil_dr'] = dr_vals.tolist()
    settings['solenoid_coil_dz'] = dz_vals.tolist()
    settings['output_directory'] = output_filename
    # Save the updated json
    save_json(settings, output_dir, output_filename_json)
