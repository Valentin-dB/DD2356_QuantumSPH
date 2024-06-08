/*
This code is a C version of the python code available at https://github.com/pmocz/QuantumSPH
It solves a simple SE problem with harmonic potential.
This is a working example of the concept in an article by Mocz and Succi (2015), https://ui.adsabs.harvard.edu/abs/2015PhRvE..91e3304M/abstract

Original author : Philip Mocz (2017), Harvard University

i d_t psi + nabla^2/2 psi -x^2 psi/2 = 0
Domain: [-inf,inf]
Potential: 1/2 x^2
Initial condition: particle in SHO
(hbar = 1, M = 1)

Particularity : Initial serial C version of the python code
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief The pi constant.
 */
#define PI 3.14159265358979323846

/**
 * @brief SPH Gaussian smoothing kernel (1D).
 *
 * This function returns the weight associated with the SPH Gaussian smoothing kernel, for the specified derivative order.
 *
 * @param r distance between particles
 * @param h scaling length
 * @param deriv derivative order
 * @return weight
 */
double kernel(double r, double h, int deriv){
    switch (deriv)
    {
    case 0:
        return pow(h,-1) / sqrt(PI) * exp(-r*r/h*h);
    case 1:
        return pow(h,-3) / sqrt(PI) * exp(-r*r/h*h) * (-2*r);
    case 2:
        return pow(h,-5) / sqrt(PI) * exp(-r*r/h*h) * (4*r*r - 2*h*h);
    case 3:
        return pow(h,-7) / sqrt(PI) * exp(-r*r/h*h) * (-8*r*r + 12*h*h) * r;
    default:
        return NAN;
    }
}

/**
 * @brief Compute density at each of the particle locations using smoothing kernel.
 *
 * @param rho density vector to be updated
 * @param x positions of the particles
 * @param n number of particles
 * @param m SPH particle mass
 * @param h scaling length
 */
void density(double * rho, double * x, int n, double m, double h){
    // for each particle
    for (int i=0; i < n; i++){
        rho[i] = 0;
        // accumulate contributions to the density
        for (int j=0; j < n; j++){
            // calculate contribution due to neighbors
            rho[i] += m*kernel(x[i] - x[j], h, 0);
        }
    }
}

/**
 * @brief Compute ''pressure'' at each of the particle locations using smoothing kernel.
 *
 * @param P pressure vector to be updated, P = -(1/4)*(d^2 rho /dx^2 - (d rho / dx)^2/rho)
 * @param x positions of the particles
 * @param rho density vector
 * @param n number of particles
 * @param m SPH particle mass
 * @param h scaling length
 */
void pressure(double * P, double * x, double * rho, int n, double m, double h){
    // for each particle
    for (int i=0; i < n; i++){
        // initialize variables
        P[i] = 0;
        double drho_i = 0;
        double ddrho_i = 0;
        // add the pairwise contributions to 1st, 2nd derivatives of density
        for (int j=0; j < n; j++){
            double uij = x[i] - x[j];
            // calculate contribution due to neighbors
            drho_i  += m * kernel(uij, h, 1);
            ddrho_i += m * kernel(uij, h, 2);
        }
        // add the pairwise contributions to the quantum pressure
        for (int j=0; j < n; j++){
            P[i] += 0.25 * (drho_i*drho_i / rho[i] - ddrho_i) * m / rho[i] * kernel(x[i] - x[j], h, 0);
        }
    }
}

/**
 * @brief Calculates acceleration of each particle.
 * 
 * Calculates acceleration of each particle due to quantum pressure, harmonic potential, and velocity damping.
 *
 * @param a acceleration vector to be updated
 * @param x positions of the particles
 * @param u velocity vector
 * @param rho density vector
 * @param P pressure vector
 * @param n number of particles
 * @param m SPH particle mass
 * @param b damping coefficient
 * @param h scaling length
 */
void acceleration(double * a, double * x, double * u, double * rho, double * P, int n, double m, double b, double h){
    // for each particle
    for (int i=0; i < n; i++){
        // contribution due to damping (-u*b) & harmonic potential (-d(0.5 x^2)/dx = - x)
        a[i] = - u[i]*b - x[i];
        // accumulate contributions to the acceleration due to quantum pressure (pairwise calculation)
        for (int j=0; j < n; j++){
            if(j != i){
                // calculate acceleration due to pressure
                double fac = -m * (P[i]/(rho[i]*rho[i]) + P[j]/(rho[j]*rho[j]));
                a[i] += fac * kernel(x[i] - x[j], h, 1);
            }
        }
    }
}

/**
 * @brief Probe the density at specified locations.
 *
 * @param rr density at specified locations, vector to be updated
 * @param x positions of the particles
 * @param n number of particles
 * @param m SPH particle mass
 * @param h scaling length
 * @param xx probe locations
 * @param nxx number of probe locations
 */
void probeDensity(double * rr, double * x, int n, double m, double h, double * xx, int nxx){
    // for each probe location
    for (int i=0; i < nxx; i++){
        rr[i] = 0;
        // accumulate contributions to the density
        for (int j = 0; j < n; j++){
            // calculate contribution due to neighbors
            rr[i] += m * kernel(xx[i] - x[j], h, 0);
        }
    }
}

/**
 * @brief Main loop.
 *
 * Evolve the time-dependant Schrödinger equation, and save the solution in "results.csv"
 */
int main(int argc, char* argv[]){

    clock_t start, end;
    start = clock();
    // Particle in SHO - c.f. Mocz & Succi (2015) Fig. 2
    // parameters
    int n = 200;               // number of particles
    int nxx = 400;             // number of probe locations
    double dt = 0.01;          // timestep
    int nt = 200;              // number of timesteps
    int nt_setup = 800;        // number of timesteps to set up simulation
    int n_out = 50;            // plot solution every nout steps
    double b = 4.0;            // velocity damping for acquiring initial condition
    double m = 1.0/n;          // mass of SPH particle ( m * n = 1 normalizes |wavefunction|^2 to 1)
    double h = 40.0/n;         // smoothing length
    double t = 0.0;            // time

    double dx, rr_exact;
    int i,j;

    // set up
    double * xx  = (double *) malloc(nxx*sizeof(double));
    double * rr  = (double *) malloc(nxx*sizeof(double));
    double * x   = (double *) malloc(n*sizeof(double));
    double * u   = (double *) calloc(n,sizeof(double));
    double * rho = (double *) malloc(n*sizeof(double));
    double * P   = (double *) malloc(n*sizeof(double));
    double * a   = (double *) malloc(n*sizeof(double));
    double * u_mhalf = (double *) malloc(n*sizeof(double));
    double * u_phalf = (double *) malloc(n*sizeof(double));
    FILE * file;

    // plot potential
    file = fopen("results.csv", "w");
    fprintf(file,"n,%d,linewidth,%.2f,color,%.2f,%.2f,%.2f\n",nxx,5.0,0.7,0.7,0.9);
    dx = 8.0/(nxx-1);
    for(i = 0; i < nxx; i++){
        xx[i] = -4.0 + dx*i;
        fprintf(file,"%f,%f\n",xx[i],0.5*xx[i]*xx[i]);
    }
    fclose(file);

    // initialize
    dx = 6.0/(n-1);
    for(i = 0; i < n; i++){
        x[i] = -3.0 + dx*i;
    }
    density(rho, x, n, m, h);
    pressure(P, x, rho, n, m, h);
    acceleration(a, x, u, rho, P, n, m, b, h);

    // get v at t=-0.5*dt for the leap frog integrator using Euler's method
    for(i = 0; i < n; i++){
        u_mhalf[i] = -0.5 * dt * a[i];
    }

    // main loop (time evolution)
    for (i = -nt_setup; i < nt; i++){   // negative time (t<0, i<0) is used to set up initial conditions
        // leap frog
        for (j = 0; j < n; j++){
            u_phalf[j] = u_mhalf[j] + a[j]*dt;
            x[j] += u_phalf[j]*dt;
            u[j] = 0.5*(u_mhalf[j] + u_phalf[j]);
            u_mhalf[j] = u_phalf[j];
        }
        if (i >= 0) t += dt;

        if (i == -1 ){  // switch off damping before t=0
            for(j=0; j < n; j++){
                u[j] = 1.0;
                u_mhalf[j] = 1.0;
            }
            b = 0;  // switch off damping at time t=0
        }

        // update densities, pressures, accelerations
        density(rho, x, n, m, h);
        pressure(P, x, rho, n, m, h);
        acceleration(a, x, u, rho, P, n, m, b, h);

        // plot solution every n_out steps
        if( (i >= 0) && (i % n_out) == 0 ){
            probeDensity(rr, x, n, m, h, xx, nxx);
            file = fopen("results.csv", "a");
            fprintf(file,"n,%d,linewidth,%.2f,color,%.2f,%.2f,%.2f\n",nxx,2.0,0.6,0.6,0.6);
            for (j = 0; j < nxx; j++){
                rr_exact = 1.0 / sqrt(PI) * exp(-pow(xx[j]-sin(t),2));
                fprintf(file,"%f,%f\n",xx[j],rr_exact);
            }
            fprintf(file,"n,%d,linewidth,%.2f,color,%.2f,%.2f,%.2f,label,$t=%.2f$\n",nxx,2.0,(double) i/nt,0.0,1.0 - (double) i/nt,t);
            for (j = 0; j < nxx; j++){
                fprintf(file,"%f,%f\n",xx[j],rr[j]);
            }
            fclose(file);
        }

        // plot the t<0 damping process for fun
        if( i==-nt_setup || i==-nt_setup*3/4 || i==-nt_setup/2 ){
            probeDensity(rr, x, n, m, h, xx, nxx);
            file = fopen("results.csv", "a");
            fprintf(file,"n,%d,linewidth,%.2f,color,%.2f,%.2f,%.2f\n",nxx,1.0,0.9,0.9,0.9);
            for (j = 0; j < nxx; j++){
                fprintf(file,"%f,%f\n",xx[j],rr[j]);
            }
            fclose(file);
        }
    }

    free(xx);
    free(rr);
    free(x);
    free(u);
    free(rho);
    free(P);
    free(a);
    free(u_mhalf);
    free(u_phalf);

    end = clock();
    printf("Time taken = %f seconds\n",((double) (end - start)) / CLOCKS_PER_SEC);

    return 0;
}