#!/bin/bash
#SBATCH --job-name=pyxis_test
#SBATCH --output=pyxis_test.out
#SBATCH --container-name=pyxis_test
#SBATCH --container-image=docker://ubuntu

echo "Running inside container:"
grep PRETTY /etc/os-release
