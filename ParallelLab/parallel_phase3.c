#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "utils.h"

/* Original Code:

//This code is NOT buggy, just sequential. Speed it up. 
void parallel_convolution(long img[DIM_ROW+PAD][DIM_COL+PAD][DIM_RGB], long kernel[DIM_KERNEL][DIM_KERNEL], long ***convolved_img) {
    int row, col, pixel, kernel_row, kernel_col;

    for (pixel = 0; pixel < DIM_RGB; pixel++) {
        for (col = 0; col < DIM_COL; col++) {
            for (row = 0; row < DIM_ROW; row++) {
                for (kernel_col = 0; kernel_col < DIM_KERNEL; kernel_col++) {
                    for (kernel_row = 0; kernel_row < DIM_KERNEL; kernel_row++) {
                        convolved_img[row][col][pixel] += img[row+kernel_row][col+kernel_col][pixel] * kernel[kernel_row][kernel_col];
                    }
                }
                convolved_img[row][col][pixel] /= GBLUR_NORM;
            }    
        }
    }
}
*/


//Spedup Code: 
void parallel_convolution(long img[DIM_ROW + PAD][DIM_COL + PAD][DIM_RGB], long kernel[DIM_KERNEL][DIM_KERNEL], long ***convolved_img) {
    int row, col, pixel, kernel_row, kernel_col;

    #pragma omp parallel for private(row, col, pixel, kernel_row, kernel_col) collapse(2) schedule(static)
    for (row = 0; row < DIM_ROW; row++) {
        for (col = 0; col < DIM_COL; col++) {
            for (pixel = 0; pixel < DIM_RGB; pixel++) {
                long sum = 0;

                for (kernel_row = 0; kernel_row < DIM_KERNEL; kernel_row++) {
                    for (kernel_col = 0; kernel_col < DIM_KERNEL; kernel_col++) {
                        sum += img[row + kernel_row][col + kernel_col][pixel] * kernel[kernel_row][kernel_col];
                    }
                }

                sum /= GBLUR_NORM;
                convolved_img[row][col][pixel] = sum;
            }
        }
    }
}

// Running phase 3.
// Sequential version took 12.258507 time units
// Parallel version took 0.905281 time units
// Your phase 3 results are correct.
// Your phase 3 speedup was 13.541108x.