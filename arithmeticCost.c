#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief The number of iterations.
 */
#define N 1000000

/**
 * @brief Timing kernel.
 *
 * Compute the execution time of N integer addition, N double precision multiply add, and N calls to the math 'exp' function
 */
int main(int argc, char* argv[]){
    clock_t start, end;

    int j = 0;
    double res = 1.0;
    double fac = 0.13426;
    double arr[N];
    for(int i=0; i < N; i++){
        arr[i] = (double) i/N;
    }

    start = clock();
    for(int i=0; i < N; i++){
        j += i;
    }
    end = clock();
    printf("Time taken = %f seconds\n",((double) (end - start)) / CLOCKS_PER_SEC);

    start = clock();
    for(int i=0; i < N; i++){
        res += fac*arr[i];
    }
    end = clock();
    printf("Time taken = %f seconds\n",((double) (end - start)) / CLOCKS_PER_SEC);

    start = clock();
    for(int i=0; i < N; i++){
        res += exp(arr[i]);
    }
    end = clock();
    printf("Time taken = %f seconds\n",((double) (end - start)) / CLOCKS_PER_SEC);

    printf("j = %d\n",j);
    printf("res = %f\n",res);
}