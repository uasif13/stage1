## Stage 1

This repository contains the code to implement Stage1 of the HPC Project. It implements the distributed version of BFS, WCC (leiden), and CM (Viecut). Code is in respective folders.

### File Structure

/bfs_mpi_path
Contains distributed version of bfs
To run:
cd bfs_mpi_path
mpicxx bfs_mpi.cpp -I/usr/local/include/igraph -L/usr/local/lib -ligraph -o bfs_mpi
mpirun -n <nprocs> bfs_mpi

/datasets
Contains dataset_test, dataset_mapped, dataset.tsv (original)

/leiden
Contains distributed version of leiden (WCC)
WIP

/viecut
TODO