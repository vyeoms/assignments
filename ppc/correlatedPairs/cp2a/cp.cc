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
    // Defining the blocks for ILP
    const int chunk_size = nx<40 ? nx : 40;
    const int num_chunks = nx/chunk_size;

    int i_offset, j_offset; // Offsets for row access
    double sum_i[chunk_size], sum_j[chunk_size]; // For mean of i-th and j-th row
    double sum_ij[chunk_size], sum_ii[chunk_size], sum_jj[chunk_size]; // For covariance calculations
    double fin_i, fin_j, fin_ij, fin_ii, fin_jj;
    double cov, denom; // Covariance and denominator for correlation
    double x_i, x_j; // To read data values in double precision

    for (int i = 0; i < ny; i++) {
        i_offset = i*nx;
        for (int j = 0; j <= i; j++) {
            
            fin_i = fin_j = fin_ii = fin_ij = fin_jj = 0.0;

            for (int l = 0; l < chunk_size; l++){
                sum_i[l] = sum_j[l] = sum_ij[l] = sum_ii[l] = sum_jj[l] = 0.0;
            }

            j_offset = j*nx;
            for (int k = 0; k < num_chunks; k++) {
                for (int l = 0; l < chunk_size; l++){
                    // Convert each element to double for precision
                    x_i = static_cast<double>(data[i_offset + k*chunk_size + l]);
                    x_j = static_cast<double>(data[j_offset + k*chunk_size + l]);

                    sum_i[l] += x_i;
                    sum_j[l] += x_j;
                    sum_ij[l] += x_i * x_j;
                    sum_ii[l] += x_i * x_i;
                    sum_jj[l] += x_j * x_j;
                }
            }

            // Get the data from the leftover chunk
            for (int l = num_chunks*chunk_size; l < nx; l++){
                x_i = static_cast<double>(data[i_offset + l]);
                x_j = static_cast<double>(data[j_offset + l]);

                sum_i[l%chunk_size] += x_i;
                sum_j[l%chunk_size] += x_j;
                sum_ij[l%chunk_size] += x_i * x_j;
                sum_ii[l%chunk_size] += x_i * x_i;
                sum_jj[l%chunk_size] += x_j * x_j;
            }

            for (int l = 0; l < chunk_size; l++){
                fin_i += sum_i[l];
                fin_j += sum_j[l];
                fin_ij += sum_ij[l];
                fin_ii += sum_ii[l];
                fin_jj += sum_jj[l];
            }

            cov = (fin_ij - fin_i * fin_j / nx); // cov(X,Y)=E(XY)-E(X)E(Y)
            denom = sqrt(((fin_ii - fin_i * fin_i / nx)) * ((fin_jj - fin_j * fin_j / nx)));

            // Store correlation value
            result[i + j*ny] = static_cast<float>(cov / denom);
        }
    }
}