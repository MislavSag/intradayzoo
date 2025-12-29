#!/bin/bash

#PBS -N ZOO_PREPARE_PANEL
#PBS -l mem=32GB

cd ${PBS_O_WORKDIR}
apptainer run image.sif estimation_panel.R