#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "utils.h"

/* Original Code:

//This code is buggy! Find the bug and speed it up.
void parallel_avg_pixel(long img[DIM_ROW][DIM_COL][DIM_RGB], long *avgs) {
    int row, col, pixel;
    long count = 0;

    #pragma omp parallel
    for (pixel = 0; pixel < DIM_RGB; pixel++) {
        for (col = 0; col < DIM_COL; col++) {
            for (row = 0; row < DIM_ROW; row++){
                avgs[pixel] += img[row][col][pixel];
                count++;
            }
        }
    }

    count /= 3;

    for (pixel = 0; pixel < DIM_RGB; pixel++) {
        avgs[pixel] /= count;
    }
}
*/

//Spedup Code:
void parallel_avg_pixel(long img[DIM_ROW][DIM_COL][DIM_RGB], long *avgs) {
    int row, col, pixel;
    long counts[DIM_RGB] = {0};

    #pragma omp parallel for collapse(2) reduction(+:avgs[:DIM_RGB], counts[:DIM_RGB]) schedule(static) //schedule(static) is used for proper load management
    for (pixel = 0; pixel < DIM_RGB; pixel++) {
        for (col = 0; col < DIM_COL; col++) { //collapse(2) combines the two inner loops into a single parallelized loop
            for (row = 0; row < DIM_ROW; row++) {
                avgs[pixel] += img[row][col][pixel]; //Reduction on operator + to avoid excessive use of criticals which runs sequentially
                counts[pixel]++; 
            }
        }
    }

    for (pixel = 0; pixel < DIM_RGB; pixel++) {
        avgs[pixel] /= counts[pixel]; //Reduction as well on operator / to avoid excessive use of criticals which runs sequentially
    }
}

// Running phase 1.
// Sequential version took 2.069586 time units
// Warning! If the your parallel solution seems like it is taking forever, CTRL-C. You most likely have not fixed the bug.
// Parallel version took 0.416112 time units
// Your phase 1 results are correct.
// Your phase 1 speedup was 4.973629x.