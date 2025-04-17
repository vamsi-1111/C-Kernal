#include <stdlib.h>
#include <math.h>
#include <float.h>

void kmeans_clustering(float* pixels, int num_pixels, int num_centroids,
                       int max_iters, int seed, float* centroids, int* labels) {
    srand(seed);

    for (int i = 0; i < num_centroids; ++i) {
        int rand_idx = rand() % num_pixels;
        centroids[i * 3 + 0] = pixels[rand_idx * 3 + 0];
        centroids[i * 3 + 1] = pixels[rand_idx * 3 + 1];
        centroids[i * 3 + 2] = pixels[rand_idx * 3 + 2];
    }

    float* new_centroids = (float*)calloc(num_centroids * 3, sizeof(float));
    int* counts = (int*)calloc(num_centroids, sizeof(int));

    for (int iter = 0; iter < max_iters; ++iter) {
        for (int i = 0; i < num_pixels; ++i) {
            float min_dist = FLT_MAX;
            int best = 0;
            float r = pixels[i * 3 + 0];
            float g = pixels[i * 3 + 1];
            float b = pixels[i * 3 + 2];

            for (int c = 0; c < num_centroids; ++c) {
                float cr = centroids[c * 3 + 0];
                float cg = centroids[c * 3 + 1];
                float cb = centroids[c * 3 + 2];
                float dist = (r - cr) * (r - cr) + (g - cg) * (g - cg) + (b - cb) * (b - cb);
                if (dist < min_dist) {
                    min_dist = dist;
                    best = c;
                }
            }
            labels[i] = best;
        }

        for (int i = 0; i < num_centroids * 3; ++i) new_centroids[i] = 0.0f;
        for (int i = 0; i < num_centroids; ++i) counts[i] = 0;

        for (int i = 0; i < num_pixels; ++i) {
            int label = labels[i];
            new_centroids[label * 3 + 0] += pixels[i * 3 + 0];
            new_centroids[label * 3 + 1] += pixels[i * 3 + 1];
            new_centroids[label * 3 + 2] += pixels[i * 3 + 2];
            counts[label]++;
        }

        for (int c = 0; c < num_centroids; ++c) {
            if (counts[c] > 0) {
                centroids[c * 3 + 0] = new_centroids[c * 3 + 0] / counts[c];
                centroids[c * 3 + 1] = new_centroids[c * 3 + 1] / counts[c];
                centroids[c * 3 + 2] = new_centroids[c * 3 + 2] / counts[c];
            }
        }
    }

    free(new_centroids);
    free(counts);
}

