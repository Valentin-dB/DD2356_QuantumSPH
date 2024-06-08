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

Particularity : OpenMP parallelization of the code. Cache-blocked 16*16 memorization of interactions between particles.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

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
 * @brief Allocate memory space for a matrix.
 *
 * This function allocates memory space for a matrix with n_row rows and n_col columns
 *
 * @param n_row number of rows
 * @param n_col number of columns
 * @return pointer to space allocated
 */
double ** InitializeMatrix(int n_row, int n_col){
    double ** A = (double **) malloc(n_row*sizeof(double *));
    double * a = malloc(n_row*n_col*sizeof(double));
    for (int i=0; i < n_row; i++){
        A[i] = a + i*n_col;
    }
    return A;
}

/**
 * @brief Frees memory space allocated for a matrix
 *
 * @param A matrix, pointer to memory space to be freed
 */
void FreeMatrix(double ** A){
    free(A[0]);
    free(A);
}

/**
 * @brief SPH Gaussian smoothing kernel (1D), for a set of particles.
 *
 * This function computes the weight associated with the SPH Gaussian smoothing kernel, for all pairwise interactions of the particles in x.
 * It stores them in ker.
 *
 * @param ker matrix to be updated with the computed weights
 * @param x positions of the particles
 * @param n number of particles
 */
void kernel_0_mat(double **ker, double *x, int n) {
    int n_blocks = (n+15)/16;

    #pragma omp for schedule(dynamic)
    for(int k = 0; k < n_blocks*(n_blocks+1)/2; k++){// Unrolling 2D loop in 1D to enable correct parallelism
        // Getting correct ii and jj indexes
        int ii = 0;
        int jj;
        for(jj=k; jj > n_blocks-ii-1; ii++){
            jj -= n_blocks - ii;
        }
        ii *= 16;
        jj = ii + jj*16;

        // Start actual work

        // End of row blocks
        if (jj > n-16) {
            // Last diagonal block
            if (ii == jj){
                for (int i = ii; i < n; i++) {
                    ker[i][i] = inv_h;
                    for (int j = i + 1; j < n; j++) {
                        double r = x[i] - x[j];
                        ker[i][j] = exp(-r * r * inv_h_sq) * inv_h;
                        ker[j][i] = ker[i][j];
                    }
                }
            }
            // Other end of row blocks
            else {
                for (int i = ii; i < ii + 16; i++) {
                    for (int j = jj; j < n; j++) {
                        double r = x[i] - x[j];
                        ker[i][j] = exp(-r * r * inv_h_sq) * inv_h;
                        ker[j][i] = ker[i][j];
                    }
                }
            }
        }

        // Diagonal blocks
        else if (ii == jj) {
            for (int i = ii; i < ii + 16; i++) {
                ker[i][i] = inv_h;
                for (int j = i + 1; j < ii + 16; j++) {
                    double r = x[i] - x[j];
                    ker[i][j] = exp(-r * r * inv_h_sq) * inv_h;
                    ker[j][i] = ker[i][j];
                }
            }
        }

        // Other blocks
        else {
            for (int i = ii; i < ii + 16; i++) {
                for (int j = jj; j < jj + 16; j++) {
                    double r = x[i] - x[j];
                    ker[i][j] = exp(-r * r * inv_h_sq) * inv_h;
                    ker[j][i] = ker[i][j];
                }
            }
        }
    }
}

/**
 * @brief SPH Gaussian smoothing kernel (1D), first derivative, for a set of particles.
 *
 * This function computes the weight associated with the first derivative of the SPH Gaussian smoothing kernel, for all pairwise interactions of the particles in x.
 * It stores them in ker.
 *
 * @param ker matrix to be updated with the computed weights
 * @param x positions of the particles
 * @param n number of particles
 */
void kernel_1_mat(double **ker, double *x, int n) {
    int n_blocks = (n+15)/16;

    #pragma omp for schedule(dynamic)
    for(int k = 0; k < n_blocks*(n_blocks+1)/2; k++){// Unrolling 2D loop in 1D to enable correct parallelism
        // Getting correct ii and jj indexes
        int ii = 0;
        int jj;
        for(jj=k; jj > n_blocks-ii-1; ii++){
            jj -= n_blocks - ii;
        }
        ii *= 16;
        jj = ii + jj*16;

        // Start actual work

        // End of row blocks
        if (jj > n-16) {
            // Last diagonal block
            if (ii == jj){
                for (int i = ii; i < n; i++) {
                    ker[i][i] = 0;
                    for (int j = i + 1; j < n; j++) {
                        double r = x[i] - x[j];
                        ker[i][j] = exp(-r*r*inv_h_sq) * (-2*r) * inv_h_cb;
                        ker[j][i] = -ker[i][j];
                    }
                }
            }
            // Other end of row blocks
            else {
                for (int i = ii; i < ii + 16; i++) {
                    for (int j = jj; j < n; j++) {
                        double r = x[i] - x[j];
                        ker[i][j] = exp(-r*r*inv_h_sq) * (-2*r) * inv_h_cb;
                        ker[j][i] = -ker[i][j];
                    }
                }
            }
        }

        // Diagonal blocks
        else if (ii == jj) {
            for (int i = ii; i < ii + 16; i++) {
                ker[i][i] = 0;
                for (int j = i + 1; j < ii + 16; j++) {
                    double r = x[i] - x[j];
                    ker[i][j] = exp(-r*r*inv_h_sq) * (-2*r) * inv_h_cb;
                    ker[j][i] = -ker[i][j];
                }
            }
        }

        // Other blocks
        else {
            for (int i = ii; i < ii + 16; i++) {
                for (int j = jj; j < jj + 16; j++) {
                    double r = x[i] - x[j];
                    ker[i][j] = exp(-r*r*inv_h_sq) * (-2*r) * inv_h_cb;
                    ker[j][i] = -ker[i][j];
                }
            }
        }
    }
}

/**
 * @brief SPH Gaussian smoothing kernel (1D), second derivative, for a set of particles.
 *
 * This function computes the weight associated with the second derivative of the SPH Gaussian smoothing kernel, for all pairwise interactions of the particles in x.
 * It stores them in ker.
 *
 * @param ker matrix to be updated with the computed weights
 * @param x positions of the particles
 * @param n number of particles
 */
void kernel_2_mat(double **ker, double *x, int n) {
    int n_blocks = (n+15)/16;

    #pragma omp for schedule(dynamic)
    for(int k = 0; k < n_blocks*(n_blocks+1)/2; k++){// Unrolling 2D loop in 1D to enable correct parallelism
        // Getting correct ii and jj indexes
        int ii = 0;
        int jj;
        for(jj=k; jj > n_blocks-ii-1; ii++){
            jj -= n_blocks - ii;
        }
        ii *= 16;
        jj = ii + jj*16;

        // Start actual work

        // End of row blocks
        if (jj > n-16) {
            // Last diagonal block
            if (ii == jj){
                for (int i = ii; i < n; i++) {
                    ker[i][i] = -2*inv_h_cb;
                    for (int j = i + 1; j < n; j++) {
                        double r = x[i] - x[j];
                        double r_sq = r*r;
                        ker[i][j] = exp(-r_sq*inv_h_sq) * (4*r_sq - 2*h_sq) * inv_h_5;
                        ker[j][i] = ker[i][j];
                    }
                }
            }
            // Other end of row blocks
            else {
                for (int i = ii; i < ii + 16; i++) {
                    for (int j = jj; j < n; j++) {
                        double r = x[i] - x[j];
                        double r_sq = r*r;
                        ker[i][j] = exp(-r_sq*inv_h_sq) * (4*r_sq - 2*h_sq) * inv_h_5;
                        ker[j][i] = ker[i][j];
                    }
                }
            }
        }

        // Diagonal blocks
        else if (ii == jj) {
            for (int i = ii; i < ii + 16; i++) {
                ker[i][i] = -2*inv_h_cb;
                for (int j = i + 1; j < ii + 16; j++) {
                    double r = x[i] - x[j];
                    double r_sq = r*r;
                    ker[i][j] = exp(-r_sq*inv_h_sq) * (4*r_sq - 2*h_sq) * inv_h_5;
                    ker[j][i] = ker[i][j];
                }
            }
        }

        // Other blocks
        else {
            for (int i = ii; i < ii + 16; i++) {
                for (int j = jj; j < jj + 16; j++) {
                    double r = x[i] - x[j];
                    double r_sq = r*r;
                    ker[i][j] = exp(-r_sq*inv_h_sq) * (4*r_sq - 2*h_sq) * inv_h_5;
                    ker[j][i] = ker[i][j];
                }
            }
        }
    }
}

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
 * @brief Compute ''relative pressure'' at each of the particle locations using smoothing kernel.
 *
 * @param rP pressure vector to be updated, rP = -(1/4)*(d^2 rho /dx^2 - (d rho / dx)^2/rho)/(rho^2)
 * @param x positions of the particles
 * @param ker_0 pre-computed smoothing kernel - interactions between particles
 * @param ker_1 pre-computed first derivative of smoothing kernel - interactions between particles
 * @param ker_2 pre-computed second derivative of smoothing kernel - interactions between particles
 * @param n number of particles
 */
void relativePressure(double * rP, double * x, double ** ker_0, double ** ker_1, double ** ker_2, int n){
    #pragma omp for schedule(dynamic)
    for (int i=0; i < n; i++){
        // introduce dummy variable to avoid false sharing
        double P_i = 0;
        // initialize density and it's 1st and 2nd derivatives
        double rho_i = 0;
        double drho_i = 0;
        double ddrho_i = 0;
        // add the pairwise contributions to the density and it's 1st, 2nd derivatives
        for (int j=0; j < n; j++){
            // calculate contribution due to neighbors
            rho_i   += ker_0[i][j];
            drho_i  += ker_1[i][j];
            ddrho_i += ker_2[i][j];
        }

        // compute scaling factor
        double inv_rho_i = 1.0 / rho_i;
        double fac = 0.25 * (drho_i*drho_i * inv_rho_i - ddrho_i) * inv_rho_i;
        // add the pairwise contributions to the quantum pressure
        for (int j=0; j < n; j++){
            P_i += fac * ker_0[i][j];
        }
        // update actual output
        rP[i] = P_i * inv_rho_i * inv_rho_i;
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
 * @param ker_1 pre-computed first derivative of smoothing kernel - interactions between particles
 * @param n number of particles
 * @param b damping coefficient
 */
void acceleration(double * a, double * x, double * u, double * rP, double ** ker_1, int n, double b){
    #pragma omp for schedule(dynamic)
    for (int i=0; i < n; i++){
        // introduce dummy variable to avoid false sharing
        // and add contribution due to damping (-u*b) & harmonic potential (-d(0.5 x^2)/dx = - x)
        double a_i = - u[i]*b - x[i];
        // accumulate contributions to the acceleration due to quantum pressure (pairwise calculation)
        for (int j=0; j < n; j++){
            if(j != i){
                // calculate acceleration due to pressure
                a_i -= (rP[i] + rP[j]) * ker_1[i][j];
            }
        }
        // update actual output
        a[i] = a_i;
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
    #pragma omp for schedule(dynamic)
    for (int i=0; i < nxx; i++){
        //introduce dummy variable to avoid false sharing
        double rr_i = 0;
        // add the pairwise contributions to density
        for (int j = 0; j < n; j++){
            // calculate contribution due to neighbors
            // and accumulate contributions to the density
            rr_i += kernel_0(xx[i] - x[j]);
        }
        // update actual output
        rr[i] = rr_i;
    }
}

/**
 * @brief Main loop.
 *
 * Evolve the time-dependant Schrödinger equation, and save the solution in "results.csv"
 */
int main(int argc, char* argv[]){
    
    double start, end;
    start = omp_get_wtime();
    // Particle in SHO - c.f. Mocz & Succi (2015) Fig. 2
    // parameters
    int n = 3200;               // number of particles
    int nxx = 400;             // number of probe locations
    double dt = 0.005;          // timestep
    int nt = 400;              // number of timesteps
    int nt_setup = 1600;        // number of timesteps to set up simulation
    int n_out = 100;            // plot solution every nout steps
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

    double * xx  = (double *) malloc(nxx*sizeof(double));
    double * rr  = (double *) malloc(nxx*sizeof(double));
    double * x   = (double *) malloc(n*sizeof(double));
    double * u   = (double *) calloc(n,sizeof(double));
    double * rho = (double *) malloc(n*sizeof(double));
    double * rP  = (double *) malloc(n*sizeof(double));
    double * a   = (double *) malloc(n*sizeof(double));
    double * u_mhalf = (double *) malloc(n*sizeof(double));
    double * u_phalf = (double *) malloc(n*sizeof(double));
    double ** ker_0 = InitializeMatrix(n,n);
    double ** ker_1 = InitializeMatrix(n,n);
    double ** ker_2 = InitializeMatrix(n,n);
    FILE * file;

    #pragma omp parallel
    {
        // initialize sampling points
        dx = 8.0/(nxx-1);
        #pragma omp for schedule(static)
        for(int i = 0; i < nxx; i++){
            xx[i] = -4.0 + dx*i;
        }

        // initialize particles
        dx = 6.0/(n-1);
        #pragma omp for schedule(static)
        for(int i = 0; i < n; i++){
            x[i] = -3.0 + dx*i;
        }

        // initialize density, pressure, and acceleration
        kernel_0_mat(ker_0, x, n);
        kernel_1_mat(ker_1, x, n);
        kernel_2_mat(ker_2, x, n);
        relativePressure(rP, x, ker_0, ker_1, ker_2, n);
        acceleration(a, x, u, rP, ker_1, n, b);
        
        // plot potential
        #pragma omp master
        {
            file = fopen("results.csv", "w");
            fprintf(file,"n,%d,linewidth,%.2f,color,%.2f,%.2f,%.2f\n",nxx,5.0,0.7,0.7,0.9);
            for(int i = 0; i < nxx; i++){
                fprintf(file,"%f,%f\n",xx[i],0.5*xx[i]*xx[i]);
            }
            fclose(file);
        }

        // get v at t=-0.5*dt for the leap frog integrator using Euler's method
        #pragma omp for schedule(static)
        for(int i = 0; i < n; i++){
            u_mhalf[i] = -0.5 * dt * a[i];
        }

        // main loop (time evolution)
        for(int i = -nt_setup; i < nt; i++){   // negative time (t<0, i<0) is used to set up initial conditions
            // leap frog
            #pragma omp for schedule(static)
            for (int j = 0; j < n; j++){
                u_phalf[j] = u_mhalf[j] + a[j]*dt;
                x[j] += u_phalf[j]*dt;
                u[j] = 0.5*(u_mhalf[j] + u_phalf[j]);
                u_mhalf[j] = u_phalf[j];
            }
            #pragma omp master
            if (i >= 0) t += dt;

            if (i == -1 ){  // switch off damping before t=0
                #pragma omp for schedule(static)
                for(int j=0; j < n; j++){
                    u[j] = 1.0;
                    u_mhalf[j] = 1.0;
                }
                #pragma omp master
                b = 0;  // switch off damping at time t=0
            }

            // update densities, pressures, accelerations
            kernel_0_mat(ker_0, x, n);
            kernel_1_mat(ker_1, x, n);
            kernel_2_mat(ker_2, x, n);
            relativePressure(rP, x, ker_0, ker_1, ker_2, n);
            acceleration(a, x, u, rP, ker_1, n, b);

            // plot solution every n_out steps
            if( (i >= 0) && (i % n_out) == 0 ){
                probeDensity(rr, x, n, xx, nxx);

                #pragma omp master
                {
                    file = fopen("results.csv", "a");
                    fprintf(file,"n,%d,linewidth,%.2f,color,%.2f,%.2f,%.2f\n",nxx,2.0,0.6,0.6,0.6);
                    for(int j = 0; j < nxx; j++){
                        rr_exact = INV_SQRT_PI * exp(-pow(xx[j]-sin(t),2));
                        fprintf(file,"%f,%f\n",xx[j],rr_exact);
                    }
                    fprintf(file,"n,%d,linewidth,%.2f,color,%.2f,%.2f,%.2f,label,$t=%.2f$\n",nxx,2.0,(double) i/nt,0.0,1.0 - (double) i/nt,t);
                    for(int j = 0; j < nxx; j++){
                        fprintf(file,"%f,%f\n",xx[j],rr[j]);
                    }
                    fclose(file);
                }
            }

            // plot the t<0 damping process for fun
            if( i==-nt_setup || i==-nt_setup*3/4 || i==-nt_setup/2 ){
                probeDensity(rr, x, n, xx, nxx);

                #pragma omp master
                {
                    file = fopen("results.csv", "a");
                    fprintf(file,"n,%d,linewidth,%.2f,color,%.2f,%.2f,%.2f\n",nxx,1.0,0.9,0.9,0.9);
                    for(int j = 0; j < nxx; j++){
                        fprintf(file,"%f,%f\n",xx[j],rr[j]);
                    }
                    fclose(file);
                }
            }
        }
    }

    free(xx);
    free(rr);
    free(x);
    free(u);
    free(rho);
    free(rP);
    free(a);
    free(u_mhalf);
    free(u_phalf);
    FreeMatrix(ker_0);
    FreeMatrix(ker_1);
    FreeMatrix(ker_2);

    end = omp_get_wtime();
    printf("Time taken = %f seconds\n",end - start);

    return 0;
}