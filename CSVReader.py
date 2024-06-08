# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:12:36 2024

@author: Valentin de Bassompierre
"""

import csv
import numpy as np
import matplotlib.pyplot as plt


def read_and_plot(file_name) :
    """ Read a *.csv file and plot it's contents
    Input: file name file_name
    """
    plt.figure()
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        for row in csv_reader:
            if line_count == -1:
                n = int(row[1])
                linewidth = float(row[3])
                color = [float(row[5]),float(row[6]),float(row[7])]
                if len(row) > 8 :
                    label = row[9]
                    is_label = True
                else:
                    is_label = False
                x = np.empty(n)
                y = np.empty(n)
                line_count = 0
            else:
                x[line_count] = float(row[0])
                y[line_count] = float(row[1])
                line_count += 1
                if line_count == n:
                    line_count = -1
                    if is_label: plt.plot(x, y, linewidth=linewidth, color=color, label=label)
                    else: plt.plot(x, y, linewidth=linewidth, color=color)
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$|\psi|^2$')
    plt.axis([-2, 4, 0, 0.8])
    #plt.savefig('solution.pdf', aspect = 'normal', bbox_inches='tight', pad_inches = 0)
    plt.show()
    return

if __name__ == "__main__":
    for file in ["results_py.csv","results.csv"] :
        read_and_plot(file)
    
