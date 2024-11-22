# Precipitation Grand Challenge (PGC) Plotting

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![Platform](https://img.shields.io/badge/platform-NOAA%20Hera%20%7C%20WCOSS2-lightgrey)

**Description**  
Tools, configuration, and job scheduling for visualizing verification statistics produced for the Precipitation Grand Challenge (PGC) project.

---

## Table of Contents

1. [Precipitation Grand Challenge (PGC) Plotting](#precipitation-grand-challenge-pgc-plotting)
2. [Introduction](#introduction)
3. [Features](#features)
4. [Installation](#installation)
    - [Clone the Repository](#clone-the-repository)
    - [Ensure Machine Compatibility](#ensure-machine-compatibility)
    - [Environment Setup](#environment-setup)
    - [Required Dependencies](#required-dependencies)
5. [Directory Structure](#directory-structure)
    - [drivers/](#drivers)
    - [fix/](#fix)
    - [parm/](#parm)
    - [ush/](#ush)
6. [Usage](#usage)
    - [Modify Configuration Files](#modify-configuration-files)
    - [Submit a Job](#submit-a-job)
    - [Viewing Plots](#viewing-plots)
7. [Contributing](#contributing)
8. [Acknowledgments](#acknowledgments)
9. [Contact](#contact)

---

## Introduction

The Precipitation Grand Challenge (PGC) is a goal to provide more accurate, reliable, and timely precipitation forecasts across timescales.  Achieving that goal in part requires visualizing the performance of precipitation forecasts over several years.  This repository can ingest METplus-based statistics and produce these visualizations.

---

## Features

The plotting core of this code is a set of Python scripts designed to generate and customize the PGC plots. Features of these scripts include:
1. **Plot Generation**
- Creates PGC plots using a METplus statistics archive
- Formats plot aspects automatically, like color schemes, axis scaling
2. **Customization and Flexibility**
- Supports optional customization of plot styles and parameters, such as line colors, unit conversion, and confidence intervals
- Allows users to modify all parts of the verification configuration, including evaluation period, models, field, and metric
- Users can easily create new verification jobs
3. **Robustness and Integration**
- Tested and flexible across a variety of edge cases and error cases
- Standalone plotting that can be easily attached to larger verification workflows
- Supports job scheduling 
4. **Compatibility and Accessibility**
- Works on two different NOAA machines (Hera and WCOSS2)
- Minimal dependencies
- Open-source and hosted on Github, making it easy to review, modify, or extend functionality

---

## Installation

To use the plotting scripts on either Hera or WCOSS2, follow these steps:  

1. **Clone the Repository**  
   Download the repository to your local workspace on the desired machine:  
   ```bash
   git clone https://github.com/MarcelCaron-NOAA/pgc_plotting.git
   cd pgc_plotting
   ```
2. **Ensure Machine Compatibility**

   The repository is designed to work on the following NOAA supercomputers:
   - Hera (uses the SLURM job scheduler)
   - WCOSS2 (uses the PBS job scheduler)
3. **Environment Setup**

   The repository includes preconfigured environment setup files for each machine:
   - `hera.env` for Hera
   - `wcoss2.env` for WCOSS2
   These files will automatically load the necessary modules when the corresponding driver script is run (e.g., `source hera.env` or `source wcoss2.env`).
4. **Required Dependencies**

   The scripts rely on standard NOAA computing environments. All required modules are automatically loaded via the provided `.env` files--no additional installation is needed.

That's it! You're ready to start submitting plotting jobs.  For usage instructions, refer to the **Usage** section below.

---

## Directory Structure

The repository includes the following directories and files:

### `drivers/`
- **hera.sh**: SLURM job scheduler script for running on the Hera machine.
- **wcoss2.sh**: PBS job scheduler script for running on the WCOSS2 machine.

These scripts export necessary environment variables for plotting configurations.

### `fix/`
- **logos/**: Contains company logos (`noaa.png`, `nws.png`) for inclusion in plots.

### `parm/`
- Configuration files for various metrics and plot types (e.g., `apcp24_bss_10mm_timeseries.config`, `apcp24_crps_timeseries.config`).

These files define environment variables for evaluation periods, metrics, and other plot settings.

### `ush/`
- **timeseries.py**: Main plotting script that coordinates the preprocessing, plotting, and other tasks.
- **plot_util.py**: Utility functions for axis scaling, generating datetimes, and metric calculations.
- **settings.py**: User-defined settings like line colors, evaluation periods, and toggle options.
- **df_preprocessing.py**: Preprocesses data, filters based on parameters like lead hours and models, and creates a DataFrame for plotting.
- **plotter.py**: Contains the core plotting logic and handles plot customization.
- **prune_stat_files.py**: Reduces memory load by pruning intermediate data for plotting.
- **check_variables.py**: Validates variables before proceeding with plotting.
- **time_util.py**: Time formatting functions.
- **string_template_substitution.py**: Handles template formatting.

---

## Usage

### 1. Modify Configuration Files
Configuration files in `parm/` define how the plot should be generated.  For example, to change the evaluation period, edit the `EVAL_PERIOD` variable in the relevant `.config` file (e.g., `apcp24_bss_10mm_timeseries.config`). You may also add new config files (e.g., `[field]_[metric]_timeseries.config`).
### 2. Submit a Job
Once the installation is complete, submit a plotting job using the respective driver script, passing the necessary parameters:
- For **Hera** (SLURM):
```bash
sbatch --export=field="apcp24",metric="crps",plottype="timeseries" drivers/hera.sh
```
- For **WCOSS2** (PBS):
```bash
qsub -v field="apcp24",metric="crps",plottype="timeseries" drivers/wcoss2.sh
```
The `--export` or `-v` option allows you to pass parameters such as the metric, field, and plot type.  These parameters are used to direct the job to the relevant configuration file in `parm/`.
### 3. Viewing Plots
Once the job completes, the plots will be available in the output directory specified in the configuration file.  

---

## Contributing

Contributions welcome!  To contribute:
1. Fork the repository.
2. Create a new branch:
```bash
git checkout -b feature-branch
```
3. Make your changes and commit them:
```bash
git commit -m "Add feature or fix bug"
```
4. Push to your fork:
```bash
git push origin feature-branch
```
5. Open a pull request to the develop repository

---

## Acknowledgments
- NOAA: Special thanks for providing the computational resources
- METplus Team: For their state-of-the-art verification tool
- EMC Verification Team: For their technical and scientific expertise, and for producing the statistics used to develop these scripts.

---

## Contact
For questions or feedback, please open an issue on GitHub or contact the maintainer at [marcel.caron@noaa.gov].
