#!/bin/bash

#PBS -N pead_predictions
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -J 1-0
#PBS -o logs
#PBS -j oe

cd ${PBS_O_WORKDIR}

apptainer run image_predictors.sif predictors_padobran.R
