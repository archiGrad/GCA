// volume.h: read a raw 8-bit volume and write an 8-bit grayscale PGM image.
//
//   int* vol = read_volume("aneurism.raw", 256, 256, 256);  // w * h * d voxels: x fastest, then y, then z; values 0..255; managed memory
//   write_pgm("mip.pgm", mip, 256, 256);                     // values are clamped to 0..255
//   write_pgm("slice.pgm", vol + 128 * 256 * 256, 256, 256); // one slice (z = 128)
//   cudaFree(vol);
//
// With d smaller than the number of slices in the file, only the first d slices are read.

#pragma once
#include <cstdio>
#include <vector>

inline int* read_volume(const char* path, int w, int h, int d) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("cannot open %s\n", path);
        return nullptr;
    }
    size_t n = (size_t)w * h * d;
    std::vector<unsigned char> buf(n);
    size_t got = fread(buf.data(), 1, n, f);
    fclose(f);
    if (got != n) {
        printf("%s: expected %zu bytes, read %zu\n", path, n, got);
        return nullptr;
    }
    int* vol;
    cudaMallocManaged(&vol, n * sizeof(int));
    for (size_t i = 0; i < n; i++) vol[i] = buf[i];
    return vol;
}

inline void write_pgm(const char* path, const int* img, int w, int h) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        printf("cannot open %s\n", path);
        return;
    }
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    for (int i = 0; i < w * h; i++) {
        int v = img[i] < 0 ? 0 : (img[i] > 255 ? 255 : img[i]);
        fputc(v, f);
    }
    fclose(f);
}
