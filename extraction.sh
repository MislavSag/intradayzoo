#!/bin/bash

#PBS -N EXTRACTION
#PBS -l mem=64GB

cd ${PBS_O_WORKDIR}
apptainer run image.sif extraction.R