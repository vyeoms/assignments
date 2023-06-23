#include <iostream>
#include <algorithm>
#include <vector>
#include <omp.h>

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/

float quickselect_median(std::vector<float>& values) {
    int n = values.size();
    int mid = n / 2;

    std::nth_element(values.begin(), values.begin() + mid, values.end());

    if (n % 2 == 0) {
        // If the number of elements is even, calculate the average of the two middle values
        std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
        return (values[mid] + values[mid - 1]) / 2.0f;
    } else {
        // If the number of elements is odd, return the middle value
        return values[mid];
    }
}

void mf(int ny, int nx, int hy, int hx, const float* in, float* out) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            std::vector<float> values;
            for (int j = std::max(0, y - hy); j <= std::min(ny - 1, y + hy); ++j) {
                for (int i = std::max(0, x - hx); i <= std::min(nx - 1, x + hx); ++i) {
                    values.push_back(in[i + nx * j]);
                }
            }
            out[x + nx * y] = quickselect_median(values);
        }
    }
}
