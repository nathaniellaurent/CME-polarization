# PUNCH CME Trajectory Determination 
This project is for independent research through Southwest Research Institute for the NASA PUNCH Solar Mission.

See accompanying CME_Trajectory_Poster.pdf for the description of the project.

## File Descriptions

Analysis is in the Scripts_CME_0 and Scripts_CME_0_Isolated folder.
Under both of these folders there are 3 key python analysis files



CME_Trajectory_Poster.pdf
- Description of this project, conclusions, and scientific background

Scripts_CME_0
- CME_benchmark (jupyter notebook)

    - Imports polarized and unpolarized (pb/B) images
    - Performs background subtraction
    - Plots results and outputs to files

- CME_correction (jupyter notebook)

    - Performs perspective correction calculations outputted by CME_benchmark
    - Plots results before and after correction

- helper_funcs (python file)

    - Contains functions for analytic inversion and background subtraction used in CME_benchmark

Scripts_CME_0_Isolated
- CME_benchmark (jupyter notebook)

    - Imports polarized and unpolarized (pb/B) images
    - Performs background subtraction
    - Plots results and outputs to files

- CME_correction (jupyter notebook)

    - Performs perspective correction calculations outputted by CME_benchmark
    - Plots results before and after correction

- helper_funcs (python file)

    - Contains functions for analytic inversion and background subtraction used in CME_benchmark
