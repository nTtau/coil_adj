Example inputs for the codes to run can be found in Example_Files, these are all for a tokamak 
These come from the geometry code ( except the vtk files which are generated by converting the step files into vtk) and the current values in the TF/PF csv files ( as these are dependent on the target magnetic field a customer provides) 

Example_outputs folder shows the expected output values from the codes, organised by which code they came from, using the inputs in the Example_Files folder 


Python scripts and their uses: 

    TF_coord_gen: Used to generate coordinates mapping the centre of the TF coil from the tf_coil_points.csv file from the tokamak geometry code -> for Tokamak geometries this needs running before anything else  

    Magnetic_Field_parallel.py: used to run the entire magnetic field code in parallel, calculates the field from all coils in the system (PF and TF) and then works out the overall magnetic field and output results in csv format, calls relevent codes in the correct order
                                (can be used as a guide for setting up galaxy workflow)

    Force_calc.py: Used to run calculations for force and hoop stress on different coils -> outputs 2 csv files, CoilForces.csv gives the net force/hoop stress on each coil, force_plot.csv gices the force/stress at a series of x,y,z coordinates describing the coils in the system 

    Current_est -> Used to set an initial guess for the current based on the target field, currently only set up for solenoids 

    auto_mapping.py: used to automate mapping the results from the B_tot.csv file onto the radial build geometry and save as screenshots 

Codes called within Magnetic_field_parallel.py: 

    grid_vals_gen : Generates a csv file with the x,y,z dimensions and dx,dy,dz used for the sampling grid for this run 

    parallel_loop.py: Calculates the magnetic field (using parallel processing) of a series of circular coils/current loops described by the input csv file ( "Example_files/pf_example.csv" for the example case)

    parallel_filament.py: Calculates the magnetic field using a filament approach to model the coil ( used for non circular coils). The coil is split into small sections ( using the coordinates provided either by the user or outputted from the TF_coord_gen script) and the 
                          magnetic field from each small filament calculated. These are then summed up to give the final magnetic field. The current through the coil currently comes from the TF_vals.csv ( modified by the user to set the curretn passing through the coil 
                          would later incorporate a different way of getting current, likely customer gives target magnetic field and we work backwards?) 



