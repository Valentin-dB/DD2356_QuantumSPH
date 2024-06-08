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

Particularity : MPI parallelization of the code. Memorization of interactions between particles, but no account for symmetry.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

/**
 * @brief Constant equal to 1.0/sqrt(pi)
 */
#define INV_SQRT_PI 0.56418958354775627928

// Define global variables
int n, rank, size, n_loc, rem;

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
 * @brief SPH Gaussian smoothing kernel (1D).
 *
 * This function returns the weight associated with the SPH Gaussian smoothing kernel.
 *
 * @param r distance between particles
 * @param h scaling length
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
 * Computes ''relative pressure'' at each of the particle locations using smoothing kernel.
 * Also stores interactions relative to the smoothing kernel and it's first derivative in ker_0 and ker_1.
 *
 * @param rP pressure vector to be updated, rP = -(1/4)*(d^2 rho /dx^2 - (d rho / dx)^2/rho)/(rho^2)
 * @param ker_0 smoothing kernel to be computed - interactions between particles
 * @param ker_1 first derivative of smoothing kernel to be computed - interactions between particles
 * @param x positions of the particles
 * @param x_ext allocated memory space for receiving positions of other processes's particles
 */
void relativePressure(double * rP, double ** ker_0, double ** ker_1, double * x, double * x_ext){

    // initialize density and it's 1st and 2nd derivatives
    double rho[n_loc];
    double drho[n_loc];
    double ddrho[n_loc];
    memset(rho,0,n_loc*sizeof(double));
    memset(drho,0,n_loc*sizeof(double));
    memset(ddrho,0,n_loc*sizeof(double));

    int k = 0;
    for(int origin=0; origin < size; origin++){

        // broadcast sending process's particles
        if(origin==rank) memcpy(x_ext,x,n_loc*sizeof(double));
        int n_origin = n_loc + (origin < rem) - (rank < rem);
        MPI_Bcast(x_ext, n_origin, MPI_DOUBLE, origin, MPI_COMM_WORLD);

        // add the pairwise contributions to the density and it's 1st, 2nd derivatives, for received particles
        for(int i=0; i < n_loc; i++){
            for(int j=0; j < n_origin; j++){
                double r = x[i] - x_ext[j];

                // compute and store interactions between particles
                int kpj = k+j;
                ker_0[i][kpj] = kernel_0(r);
                ker_1[i][kpj] = kernel_1(r);

                // add contribution due to neighbors
                rho[i]   += ker_0[i][kpj];
                drho[i]  += ker_1[i][kpj];
                ddrho[i] += kernel_2(r);
            }
        }
        k += n_origin;
    }

    // initialize relative pressure vector
    memset(rP,0,n_loc*sizeof(double));
    for(int i=0; i < n_loc; i++){
        // compute scaling factor
        double inv_rho_i = 1.0 / rho[i];
        double fac = 0.25 * (drho[i]*drho[i] * inv_rho_i - ddrho[i]) * inv_rho_i;
        // add the pairwise contributions to the quantum pressure
        for(int j=0; j < n; j++){
            rP[i] += fac * ker_0[i][j];
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
 * @param rP_ext allocated memory space for receiving other process's relative pressure vectors
 * @param ker_1 pre-computed first derivative of smoothing kernel - interactions between particles
 * @param b damping coefficient
 */
void acceleration(double * a, double * x, double * u, double * rP, double * rP_ext, double ** ker_1, double b){

    // add contribution due to damping (-u*b) & harmonic potential (-d(0.5 x^2)/dx = - x)
    for(int i=0; i < n_loc; i++){
        a[i] = - u[i]*b - x[i];
    }

    int k = 0;
    for(int origin=0; origin < size; origin++){

        // broadcast sending process's relative pressure vector
        if(origin==rank) memcpy(rP_ext,rP,n_loc*sizeof(double));
        int n_origin = n_loc + (origin < rem) - (rank < rem);
        MPI_Bcast(rP_ext, n_origin, MPI_DOUBLE, origin, MPI_COMM_WORLD);

        // accumulate contributions to the acceleration due to quantum pressure, for received particles (pairwise calculation)
        for(int i=0; i < n_loc; i++){
            for(int j=0; j < n_origin; j++){
                if(j != i || origin != rank){
                    a[i] -= (rP[i] + rP_ext[j]) * ker_1[i][k+j];
                }
            }
        }
        k += n_origin;
    }
}

/**
 * @brief Probe the density at specified locations.
 *
 * @param rr density at specified locations, vector to be updated
 * @param x positions of the particles
 * @param x_ext allocated memory space for receiving positions of other processes's particles
 * @param xx probe locations
 * @param nxx number of probe locations
 */
void probeDensity(double * rr, double * x, double * x_ext, double * xx, int nxx_loc){

    // initialize probe density vector
    memset(rr,0,nxx_loc*sizeof(double));

    for(int origin=0; origin < size; origin++){

        // broadcast sending process's particles
        if(origin==rank) memcpy(x_ext,x,n_loc*sizeof(double));
        int n_origin = n_loc + (origin < rem) - (rank < rem);
        MPI_Bcast(x_ext, n_origin, MPI_DOUBLE, origin, MPI_COMM_WORLD);

        // add the pairwise contributions to the density, for received particles
        for(int i=0; i < nxx_loc; i++){
            for(int j=0; j < n_origin; j++){
                rr[i] += kernel_0(xx[i] - x_ext[j]);
            }
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
    n = 800;                   // number of particles
    int nxx = 400;             // number of probe locations
    double dt = 0.005;         // timestep
    int nt = 400;              // number of timesteps
    int nt_setup = 1600;       // number of timesteps to set up simulation
    int n_out = 100;           // plot solution every nout steps
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

    int provided, nxx_loc, rem_xx;
    double dx, rr_exact;
    int i,j;

    // initialize processes
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    rem = n%size;
    n_loc = n/size + (rank < rem);

    rem_xx = nxx%size;
    nxx_loc = nxx/size + (rank < rem_xx);

    // set up
    double * xx  = (rank == 0) ? (double *) malloc(nxx*sizeof(double)) : (double *) malloc(nxx_loc*sizeof(double));
    double * rr  = (rank == 0) ? (double *) malloc(nxx_loc*size*sizeof(double)) : (double *) malloc(nxx_loc*sizeof(double));
    double * x   = (double *) malloc(n_loc*sizeof(double));
    double * u   = (double *) calloc(n_loc,sizeof(double));
    double * a   = (double *) malloc(n_loc*sizeof(double));
    double * rP  = (double *) malloc(n_loc*sizeof(double));
    double * u_mhalf = (double *) malloc(n_loc*sizeof(double));
    double * u_phalf = (double *) malloc(n_loc*sizeof(double));
    double * x_ext   = (double *) malloc((n_loc+1)*sizeof(double));
    double * rP_ext  = (double *) malloc((n_loc+1)*sizeof(double));
    double ** ker_0  = InitializeMatrix(n_loc,n);
    double ** ker_1  = InitializeMatrix(n_loc,n);
    FILE * file;

    // initialize xx and plot potential
    dx = 8.0/(nxx-1);
    if(rank == 0) {
        file = fopen("results.csv", "w");
        fprintf(file,"n,%d,linewidth,%.2f,color,%.2f,%.2f,%.2f\n",nxx,5.0,0.7,0.7,0.9);
        for(i = 0; i < nxx; i++){
            xx[i] = -4.0 + dx*i;
            fprintf(file,"%f,%f\n",xx[i],0.5*xx[i]*xx[i]);
        }
        fclose(file);
    } else if(rank < rem_xx){
        j = 0;
        for(i = rank*nxx_loc; i < (rank+1)*nxx_loc; i++){
            xx[j] = -4.0 + dx*i;
            j++;
        }
    } else {
        j = 0;
        for(i = rank*nxx_loc + rem_xx; i < (rank+1)*nxx_loc + rem_xx; i++){
            xx[j] = -4.0 + dx*i;
            j++;
        }
    }

    // initialize x, rP, a
    dx = 6.0/(n-1);
    j = 0;
    if(rank < rem){
        for(i = rank*n_loc; i < (rank+1)*n_loc; i++){
            x[j] = -3.0 + dx*i;
            j++;
        }
    } else {
        for(i = rank*n_loc + rem; i < (rank+1)*n_loc + rem; i++){
            x[j] = -3.0 + dx*i;
            j++;
        }
    }
    relativePressure(rP, ker_0, ker_1, x, x_ext);
    acceleration(a, x, u, rP, rP_ext, ker_1, b);

    // get v at t=-0.5*dt for the leap frog integrator using Euler's method
    for(i = 0; i < n_loc; i++){
        u_mhalf[i] = -0.5 * dt * a[i];
    }

    // main loop (time evolution)
    for (i = -nt_setup; i < nt; i++){   // negative time (t<0, i<0) is used to set up initial conditions
        // leap frog
        for (j = 0; j < n_loc; j++){
            u_phalf[j] = u_mhalf[j] + a[j]*dt;
            x[j] += u_phalf[j]*dt;
            u[j] = 0.5*(u_mhalf[j] + u_phalf[j]);
            u_mhalf[j] = u_phalf[j];
        }
        if (i >= 0) t += dt;

        if (i == -1 ){  // switch off damping before t=0
            for(j=0; j < n_loc; j++){
                u[j] = 1.0;
                u_mhalf[j] = 1.0;
            }
            b = 0;  // switch off damping at time t=0
        }

        // update densities, pressures, accelerations
        relativePressure(rP, ker_0, ker_1, x, x_ext);
        acceleration(a, x, u, rP, rP_ext, ker_1, b);

        // plot solution every n_out steps
        if( (i >= 0) && (i % n_out) == 0 ){
            probeDensity(rr, x, x_ext, xx, nxx_loc);
            if(rank == 0) MPI_Gather(MPI_IN_PLACE, nxx_loc, MPI_DOUBLE, rr, nxx_loc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            else MPI_Gather(rr, nxx_loc + (0 < rem_xx) - (rank < rem_xx), MPI_DOUBLE, NULL, nxx_loc + (0 < rem_xx) - (rank < rem_xx), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if(rank == 0){
                file = fopen("results.csv", "a");
                fprintf(file,"n,%d,linewidth,%.2f,color,%.2f,%.2f,%.2f\n",nxx,2.0,0.6,0.6,0.6);
                for (j = 0; j < nxx; j++){
                    rr_exact = INV_SQRT_PI * exp(-pow(xx[j]-sin(t),2));
                    fprintf(file,"%f,%f\n",xx[j],rr_exact);
                }
                fprintf(file,"n,%d,linewidth,%.2f,color,%.2f,%.2f,%.2f,label,$t=%.2f$\n",nxx,2.0,(double) i/nt,0.0,1.0 - (double) i/nt,t);
                int index = 0;
                for (int k = 0; k < size; k++){
                    for (j = 0; j < nxx_loc + (k < rem_xx) - (0 < rem_xx); j++){
                        fprintf(file,"%f,%f\n",xx[index],rr[k*nxx_loc+j]);
                        index++;
                    }
                }
                fclose(file);
            }
        }

        // plot the t<0 damping process for fun
        if( i==-nt_setup || i==-nt_setup*3/4 || i==-nt_setup/2 ){
            probeDensity(rr, x, x_ext, xx, nxx_loc);
            if(rank == 0) MPI_Gather(MPI_IN_PLACE, nxx_loc, MPI_DOUBLE, rr, nxx_loc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            else MPI_Gather(rr, nxx_loc + (0 < rem_xx) - (rank < rem_xx), MPI_DOUBLE, rr, nxx_loc + (0 < rem_xx) - (rank < rem_xx), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if(rank == 0){
                file = fopen("results.csv", "a");
                fprintf(file,"n,%d,linewidth,%.2f,color,%.2f,%.2f,%.2f\n",nxx,1.0,0.9,0.9,0.9);
                int index = 0;
                for (int k = 0; k < size; k++){
                    for (j = 0; j < nxx_loc + (k < rem_xx) - (0 < rem_xx); j++){
                        fprintf(file,"%f,%f\n",xx[index],rr[k*nxx_loc+j]);
                        index++;
                    }
                }
                fclose(file);
            }
        }
    }

    free(xx);
    free(rr);
    free(x);
    free(u);
    free(a);
    free(rP);
    free(x_ext);
    free(rP_ext);
    free(u_mhalf);
    free(u_phalf);
    FreeMatrix(ker_0);
    FreeMatrix(ker_1);

    MPI_Finalize();

    end = clock();
    printf("Time taken = %f seconds\n",((double) (end - start)) / CLOCKS_PER_SEC);

    return 0;
}