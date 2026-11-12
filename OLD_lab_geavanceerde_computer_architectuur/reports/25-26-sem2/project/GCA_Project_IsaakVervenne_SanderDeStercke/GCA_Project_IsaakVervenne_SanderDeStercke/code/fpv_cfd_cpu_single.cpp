// Geavanceerde Computerarchitectuur Project
// CPU single-core referentie: 2D LBM (D2Q9)

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>

constexpr int Q  = 9;
constexpr int NX = 512;
constexpr int NY = 256;
constexpr int ITERATIONS = 12000;

constexpr float U_IN  = 0.04f;
constexpr float OMEGA = 1.0f;

constexpr float W[Q] = {
    4.f/9.f,
    1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f,
    1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f
};

constexpr int CX[Q]  = {0, 1, 0,-1, 0, 1,-1,-1, 1};
constexpr int CY[Q]  = {0, 0, 1, 0,-1, 1, 1,-1,-1};
constexpr int OPP[Q] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

inline float feq(int i, float rho, float ux, float uy) {
    const float cu = 3.f * (CX[i] * ux + CY[i] * uy);
    const float uu = ux * ux + uy * uy;
    return W[i] * rho * (1.f + cu + 0.5f * cu * cu - 1.5f * uu);
}

static void init_obstacle(std::vector<unsigned char>& obstacle, int nx, int ny) {
    std::fill(obstacle.begin(), obstacle.end(), 0);

    const int x0 = nx / 4;
    const int x1 = x0 + 26;
    const int yc = ny / 2;
    const int hh = 34;
    const int r  = 6;

    for (int y = 1; y < ny - 1; ++y) {
        for (int x = 1; x < nx - 1; ++x) {
            bool insideRect = (x >= x0 + r && x <= x1 - r && y >= yc - hh && y <= yc + hh);
            bool insideCaps = false;

            float dxL = float(x - (x0 + r));
            float dxR = float(x - (x1 - r));
            float dy  = float(y - yc);
            if (dxL * dxL + dy * dy <= float((hh + r) * (hh + r)) && x < x0 + r) insideCaps = true;
            if (dxR * dxR + dy * dy <= float((hh + r) * (hh + r)) && x > x1 - r) insideCaps = true;

            if (insideRect || insideCaps) obstacle[y * nx + x] = 1;
        }
    }
}

static void write_velocity_csv(const std::string& path, const std::vector<float>& vel, int nx, int ny) {
    std::ofstream out(path);
    out << "x,y,vel\n";
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            out << x << ',' << y << ',' << vel[y * nx + x] << '\n';
        }
    }
}

int main() {
    const int cells = NX * NY;

    std::vector<float> fA(cells * Q), fB(cells * Q), vel(cells, 0.f);
    std::vector<unsigned char> obstacle(cells, 0);

    init_obstacle(obstacle, NX, NY);

    for (int idx = 0; idx < cells; ++idx) {
        for (int i = 0; i < Q; ++i) {
            fA[idx * Q + i] = W[i];
            fB[idx * Q + i] = W[i];
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < ITERATIONS; ++it) {
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                const int idx = y * NX + x;

                if (y == 0 || y == NY - 1) {
                    for (int i = 0; i < Q; ++i) fB[idx * Q + i] = fA[idx * Q + OPP[i]];
                    continue;
                }

                if (obstacle[idx]) {
                    for (int i = 0; i < Q; ++i) fB[idx * Q + i] = fA[idx * Q + OPP[i]];
                    continue;
                }

                float fLoc[Q];

                if (x == 0) {
                    const float rho = 1.0f;
                    for (int i = 0; i < Q; ++i) fLoc[i] = feq(i, rho, U_IN, 0.0f);
                } else if (x == NX - 1) {
                    const int left = idx - 1;
                    for (int i = 0; i < Q; ++i) fLoc[i] = fA[left * Q + i];
                } else {
                    for (int i = 0; i < Q; ++i) {
                        const int sx = x - CX[i];
                        const int sy = y - CY[i];
                        const int sidx = sy * NX + sx;
                        fLoc[i] = obstacle[sidx] ? fA[idx * Q + OPP[i]] : fA[sidx * Q + i];
                    }
                }

                float rho = 0.f, ux = 0.f, uy = 0.f;
                for (int i = 0; i < Q; ++i) {
                    rho += fLoc[i];
                    ux  += fLoc[i] * CX[i];
                    uy  += fLoc[i] * CY[i];
                }

                if (!std::isfinite(rho) || rho < 1e-8f) {
                    rho = 1.0f; ux = 0.0f; uy = 0.0f;
                } else {
                    ux /= rho; uy /= rho;
                }

                for (int i = 0; i < Q; ++i) {
                    const float feq_i = feq(i, rho, ux, uy);
                    const float out = fLoc[i] - OMEGA * (fLoc[i] - feq_i);
                    fB[idx * Q + i] = std::isfinite(out) ? out : feq_i;
                }
            }
        }
        std::swap(fA, fB);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            const int idx = y * NX + x;
            if (obstacle[idx]) { vel[idx] = 0.0f; continue; }

            float rho = 0.f, ux = 0.f, uy = 0.f;
            for (int i = 0; i < Q; ++i) {
                const float fi = fA[idx * Q + i];
                rho += fi; ux += fi * CX[i]; uy += fi * CY[i];
            }

            if (!std::isfinite(rho) || rho < 1e-8f) {
                vel[idx] = 0.0f; continue;
            }

            ux /= rho; uy /= rho;
            const float v = std::sqrt(ux * ux + uy * uy);
            vel[idx] = std::isfinite(v) ? v : 0.0f;
        }
    }

    write_velocity_csv("velocity_cpu_single.csv", vel, NX, NY);

    std::cout << "CPU single-core simulatie klaar. Tijd: " << dt.count() << " s\n";
    std::cout << "Output geschreven: velocity_cpu_single.csv\n";

    return 0;
}
