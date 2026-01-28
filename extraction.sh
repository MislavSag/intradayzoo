#!/bin/bash

#PBS -N EXTRACTION
#PBS -l mem=150GB

cd ${PBS_O_WORKDIR}
apptainer run image.sif extraction.R