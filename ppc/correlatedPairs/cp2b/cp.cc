#include <cmath>
#include <omp.h>

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float* data, float* result) {
    // Calculate correlations between every pair of input vectors
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double sum_i = 0.0, sum_j = 0.0, sum_ij = 0.0;
            double sum_sq_i = 0.0, sum_sq_j = 0.0;
            double correlation;

            // Calculate the sums needed for correlation calculation
            for (int k = 0; k < nx; k++) {
                double value_i = static_cast<double>(data[k + i * nx]);
                double value_j = static_cast<double>(data[k + j * nx]);
                sum_i += value_i;
                sum_j += value_j;
                sum_ij += value_i * value_j;
                sum_sq_i += value_i * value_i;
                sum_sq_j += value_j * value_j;
            }

            // Calculate the correlation coefficient
            correlation = (sum_ij - (sum_i * sum_j) / nx) /
                          std::sqrt((sum_sq_i - (sum_i * sum_i) / nx) *
                                    (sum_sq_j - (sum_j * sum_j) / nx));

            result[i + j * ny] = static_cast<float>(correlation);
        }
    }
}
