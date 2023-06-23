#include <cmath>

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float* data, float* result) {
    double sum_i, sum_j; // For mean of i-th and j-th row
    double sum_ij, sum_ii, sum_jj; // For covariance calculations
    double cov, denom; // Covariance and denominator for correlation
    double x_i, x_j; // To read data values in double precision

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            sum_i = 0.0;
            sum_j = 0.0;
            sum_ij = 0.0;
            sum_ii = 0.0;
            sum_jj = 0.0;
            for (int k = 0; k < nx; k++) {
                // Convert each element to double for precision
                x_i = static_cast<double>(data[i*nx + k]);
                x_j = static_cast<double>(data[j*nx + k]);

                sum_i += x_i;
                sum_j += x_j;
                sum_ij += x_i * x_j;
                sum_ii += x_i * x_i;
                sum_jj += x_j * x_j;
            }
            
            cov = (sum_ij - sum_i * sum_j / nx); // cov(X,Y)=E(XY)-E(X)E(Y)
            denom = sqrt(((sum_ii - sum_i * sum_i / nx)) * ((sum_jj - sum_j * sum_j / nx)));

            // Store correlation value
            result[i + j*ny] = static_cast<float>(cov / denom);
        }
    }
}