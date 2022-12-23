#include <iostream>

struct Result {
    float avg[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- horizontal position: 0 <= x0 < x1 <= nx
- vertical position: 0 <= y0 < y1 <= ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
- output: avg[c]
*/
Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1, int x1) {
    Result result{{0.0f, 0.0f, 0.0f}};

    int it_x, it_y;

    int total_pixels = (x1 - x0) * (y1 - y0);

    double avg_channel[3] = {0.0, 0.0, 0.0};

    for (it_x = x0; it_x < x1; it_x++){
        for (it_y = y0; it_y < y1; it_y++){
            avg_channel[0] += data[3*it_x + 3*nx*it_y];
            avg_channel[1] += data[1 + 3*it_x + 3*nx*it_y];
            avg_channel[2] += data[2 + 3*it_x + 3*nx*it_y];
        }
    }

    result.avg[0] = avg_channel[0] / total_pixels;
    result.avg[1] = avg_channel[1] / total_pixels;
    result.avg[2] = avg_channel[2] / total_pixels;

    return result;
}
