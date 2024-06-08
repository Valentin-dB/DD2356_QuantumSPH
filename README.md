# DD2356_QuantumSPH

Valentin de Bassompierre (2024)

This repository contains multiple C versions of the python code available at https://github.com/pmocz/QuantumSPH (and present in this repository under quantumsph.py)\n
All codes solve a simple SE problem with harmonic potential.\n
This is a working example of the concept in an article by Mocz and Succi (2015), https://ui.adsabs.harvard.edu/abs/2015PhRvE..91e3304M/abstract

Original author : Philip Mocz (2017), Harvard University

This project is part of the course DD2356 Methods in High-Performance Computing given at the KTH Royal Institute of Technology.\n
The different C versions of the code represent various optimizations and avenues explored by the author during the course of the project.\n
To fully understand the differences between each version of the code, please read the project report: MHPC___Project.pdf

Note that all versions of the code produce the same output, provided the same settings are given (number of particles, time step, number of time steps...).\n
Please note that these settings may not be set to the same values by default for every version of the code.\n
Once run, the C codes produce a file called "results.csv". This file may be read by running "CSVReader.py", which produces a graph from the results.\n
These graphs can be compared to those produced by the original python code.\n
For testing purposes, one might also wish to directly compare the CSV output, "results.csv", to that of the original python code, "results_py.csv".\n
It is possible to run the original python code with specific settings to produce a CSV output file and generate graphs for these settings.

Please find the documentation for the code in: Documentation.pdf
