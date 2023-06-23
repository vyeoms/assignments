#include <algorithm>
#include <vector>

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/

void mf(int ny, int nx, int hy, int hx, const float* in, float* out) {
    int windowSize = (2*hx+1) * (2*hy+1);
    std::vector<float> window(windowSize);

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            // Count how many pixels are inside the window
            int count = 0;

            for (int j = y - hy; j <= y + hy; j++) {
                for (int i = x - hx; i <= x + hx; i++) {
                    if (i >= 0 && i < nx && j >= 0 && j < ny) {
                        window[count] = in[i + j*nx];// Only apply the filter over the pixels present in the window
                        count++;// Add to pixel count
                    }
                }
            }

            // Standard mediann
            int medianIndex = count / 2;
            std::nth_element(window.begin(), window.begin() + medianIndex, window.begin() + count);
            float median = window[medianIndex];

            // In case there was an even number of pixels apply median rule
            if (count%2 == 0){
                std::nth_element(window.begin(), window.begin() + medianIndex - 1, window.begin() + count);
                median += window[medianIndex-1];
                median *= 0.5;
            }

            out[x + y*nx] = median;
        }
    }
}