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

Particularity : Improved serial version of the code, without memorization
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief Constant equal to 1.0/sqrt(pi)
 */
#define INV_SQRT_PI 0.56418958354775627928

// Define reusable constants
double h_sq;
double inv_h;     // * m / sqrt(pi)
double inv_h_sq;
double inv_h_cb;  // * m / sqrt(pi)
double inv_h_5;   // * m / sqrt(pi)
double inv_h_7;   // * m / sqrt(pi)

/**
 * @brief SPH Gaussian smoothing kernel (1D).
 *
 * This function returns the weight associated with the SPH Gaussian smoothing kernel.
 *
 * @param r distance between particles
 * @return weight
 */
double kernel_0(double r){
    return exp(-r*r*inv_h_sq) * inv_h;
}

/**
 * @brief SPH Gaussian smoothing kernel (1D), first derivative.
 *
 * This function returns the weight associated with the first derivative of the SPH Gaussian smoothing kernel.
 *
 * @param r distance between particles
 * @return weight
 */
double kernel_1(double r){
    return exp(-r*r*inv_h_sq) * (-2*r) * inv_h_cb;
}

/**
 * @brief SPH Gaussian smoothing kernel (1D), second derivative.
 *
 * This function returns the weight associated with the second derivative of the SPH Gaussian smoothing kernel.
 *
 * @param r distance between particles
 * @return weight
 */
double kernel_2(double r){
    double r_sq = r*r;
    return exp(-r_sq*inv_h_sq) * (4*r_sq - 2*h_sq) * inv_h_5;
}

/**
 * @brief Compute ''relative pressure'' at each of the particle locations using smoothing kernel.
 *
 * @param rP pressure vector to be updated, rP = -(1/4)*(d^2 rho /dx^2 - (d rho / dx)^2/rho)/(rho^2)
 * @param x positions of the particles
 * @param n number of particles
 */
void pressure(double * rP, double * x, int n){
    for (int i=0; i < n; i++){
        rP[i] = 0;
        // initialize density and it's 1st and 2nd derivatives
        double rho_i = 0;
        double drho_i = 0;
        double ddrho_i = 0;
        // add the pairwise contributions to the density and it's 1st, 2nd derivatives
        for (int j=0; j < n; j++){
            double uij = x[i] - x[j];
            // calculate contribution due to neighbors
            rho_i   += kernel_0(uij);
            drho_i  += kernel_1(uij);
            ddrho_i += kernel_2(uij);
        }

        // compute scaling factor
        double inv_rho_i = 1.0 / rho_i;
        double fac = 0.25 * (drho_i*drho_i * inv_rho_i - ddrho_i) * inv_rho_i;
        // add the pairwise contributions to the quantum pressure
        for (int j=0; j < n; j++){
            rP[i] += fac * kernel_0(x[i] - x[j]);
        }
        // scale the pressure with respect to the density
        rP[i] *= inv_rho_i * inv_rho_i;
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
 * @param rP relative pressure vector
 * @param n number of particles
 * @param b damping coefficient
 */
void acceleration(double * a, double * x, double * u, double * rP, int n, double b){
    for (int i=0; i < n; i++){
        // contribution due to damping (-u*b) & harmonic potential (-d(0.5 x^2)/dx = - x)
        a[i] = - u[i]*b - x[i];
        // accumulate contributions to the acceleration due to quantum pressure (pairwise calculation)
        for (int j=0; j < n; j++){
            if(j != i){
                // calculate acceleration due to pressure
                double fac = rP[i] + rP[j];
                a[i] -= fac * kernel_1(x[i] - x[j]);
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
 * @param xx probe locations
 * @param nxx number of probe locations
 */
void probeDensity(double * rr, double * x, int n, double * xx, int nxx){
    for (int i=0; i < nxx; i++){
        rr[i] = 0;
        // add the pairwise contributions to density
        for (int j = 0; j < n; j++){
            // calculate contribution due to neighbors
            // and accumulate contributions to the density
            rr[i] += kernel_0(xx[i] - x[j]);
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
    int n = 100;               // number of particles
    int nxx = 400;             // number of probe locations
    double dt = 0.02;          // timestep
    int nt = 100;              // number of timesteps
    int nt_setup = 400;        // number of timesteps to set up simulation
    int n_out = 25;            // plot solution every nout steps
    double b = 4.0;            // velocity damping for acquiring initial condition
    double m = 1.0/n;          // mass of SPH particle ( m * n = 1 normalizes |wavefunction|^2 to 1)
    double h = 40.0/n;         // smoothing length
    double t = 0.0;            // time

    // initialize global variables to be re-used
    h_sq = h*h;
    inv_h = 1.0 / h;
    inv_h_sq = inv_h * inv_h;
    inv_h_cb = inv_h * inv_h_sq;
    inv_h_5 = inv_h_cb * inv_h_sq;
    inv_h_7 = inv_h_5 * inv_h_sq;

    inv_h = inv_h * m * INV_SQRT_PI;
    inv_h_cb = inv_h_cb * m * INV_SQRT_PI;
    inv_h_5 = inv_h_5 * m * INV_SQRT_PI;
    inv_h_7 = inv_h_7 * m * INV_SQRT_PI;

    // set up
    double dx, rr_exact;
    int i,j;

    double * xx  = (double *) malloc(nxx*sizeof(double));
    double * rr  = (double *) malloc(nxx*sizeof(double));
    double * x   = (double *) malloc(n*sizeof(double));
    double * u   = (double *) calloc(n,sizeof(double));
    double * rP  = (double *) malloc(n*sizeof(double));
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

    // initialize particles locations, relative pressure and acceleration
    dx = 6.0/(n-1);
    for(i = 0; i < n; i++){
        x[i] = -3.0 + dx*i;
    }
    pressure(rP, x, n);
    acceleration(a, x, u, rP, n, b);

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
        pressure(rP, x, n);
        acceleration(a, x, u, rP, n, b);
        // plot solution every n_out steps
        if( (i >= 0) && (i % n_out) == 0 ){
            probeDensity(rr, x, n, xx, nxx);
            file = fopen("results.csv", "a");
            fprintf(file,"n,%d,linewidth,%.2f,color,%.2f,%.2f,%.2f\n",nxx,2.0,0.6,0.6,0.6);
            for (j = 0; j < nxx; j++){
                rr_exact = INV_SQRT_PI * exp(-pow(xx[j]-sin(t),2));
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
            probeDensity(rr, x, n, xx, nxx);
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
    free(rP);
    free(a);
    free(u_mhalf);
    free(u_phalf);

    end = clock();
    printf("Time taken = %f seconds\n",((double) (end - start)) / CLOCKS_PER_SEC);

    return 0;
}