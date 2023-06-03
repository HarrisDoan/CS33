#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "utils.h"

/* Original Code:

//This code is buggy! Find the bug and speed it up.
void parallel_to_grayscale(long img[DIM_ROW][DIM_COL][DIM_RGB], long ***grayscale_img, long *min_max_gray) {
    int row, col, pixel, gray_pixel;
    int min_gray = 256;
    int max_gray = -1;

    #pragma omp parallel for private(row, gray_pixel, pixel)
    for (col = 0; col < DIM_COL; col++) {
        for (row = 0; row < DIM_ROW; row++){
            for (gray_pixel = 0; gray_pixel < DIM_RGB; gray_pixel++) {
                for (pixel = 0; pixel < DIM_RGB; pixel++) {
                    grayscale_img[row][col][gray_pixel] += img[row][col][pixel];
                }
                grayscale_img[row][col][gray_pixel] /= 3;
                if (grayscale_img[row][col][gray_pixel] < min_gray) {
                    min_max_gray[0] = grayscale_img[row][col][gray_pixel];
                    min_gray = grayscale_img[row][col][gray_pixel];
                }
                if (grayscale_img[row][col][gray_pixel] > max_gray) {
                    min_max_gray[1] = grayscale_img[row][col][gray_pixel];
                    max_gray = grayscale_img[row][col][gray_pixel];
                }
            }
        }
    }
}
*/


//Spedup Code:
void parallel_to_grayscale(long img[DIM_ROW][DIM_COL][DIM_RGB], long ***grayscale_img, long *min_max_gray) {
    int row, col, pixel, gray_pixel;
    int min_gray = 256;
    int max_gray = -1;

    #pragma omp parallel for collapse(2) private(gray_pixel) reduction(min:min_gray) reduction(max:max_gray) schedule(static)
    for (row = 0; row < DIM_ROW; row++) { //swap the col loop and row loop from the original code
        for (col = 0; col < DIM_COL; col++) {
            for (gray_pixel = 0; gray_pixel < DIM_RGB; gray_pixel++) { //private the gray_pixel variable
                long sum = 0; //Create a variable called sum instead of doing: grayscale_img[row][col][gray_pixel] += img[row][col][pixel];
                for (pixel = 0; pixel < DIM_RGB; pixel++) {
                    sum += img[row][col][pixel];
                }
                grayscale_img[row][col][gray_pixel] = sum / DIM_RGB;

                if (grayscale_img[row][col][gray_pixel] < min_gray) {
                    min_gray = grayscale_img[row][col][gray_pixel];
                }

                if (grayscale_img[row][col][gray_pixel] > max_gray) {
                    max_gray = grayscale_img[row][col][gray_pixel];
                }
            }
        }
    }

    min_max_gray[0] = min_gray;
    min_max_gray[1] = max_gray;
}

// Running phase 2.
// Sequential version took 6.904924 time units
// Parallel version took 0.873016 time units
// Your phase 2 results are correct. However, they may only be correct for this run, make sure to take care of possible data races!
// Your phase 2 speedup was 7.909273x.