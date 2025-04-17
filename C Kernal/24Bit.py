import numpy as np
from PIL import Image
import argparse
import ctypes
import time
import os

lib = ctypes.CDLL("./libkernel.so")

lib.kmeans_clustering.argtypes = [
    ctypes.POINTER(ctypes.c_float),  
    ctypes.c_int,                    
    ctypes.c_int,                    
    ctypes.c_int,                    
    ctypes.c_int,                    
    ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_int)   
]

def image_to_array(image_path):
    img = Image.open(image_path).convert('RGB')
    data = np.asarray(img).astype(np.float32)
    return data, img.size, img

def flatten_image(img_data):
    return img_data.reshape(-1, 3)

def run_python_kmeans(pixels, num_centroids, max_iters, seed):
    np.random.seed(seed)
    centroids = pixels[np.random.choice(len(pixels), num_centroids, replace=False)].astype(np.float32)
    for _ in range(max_iters):
        distances = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        for i in range(num_centroids):
            if np.any(labels == i):
                centroids[i] = np.mean(pixels[labels == i], axis=0)
    return centroids, labels

def run_c_kmeans(pixels, num_centroids, max_iters, seed):
    num_pixels = pixels.shape[0]
    pixel_ptr = pixels.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    centroids = np.zeros((num_centroids, 3), dtype=np.float32)
    centroids_ptr = centroids.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    labels = np.zeros(num_pixels, dtype=np.int32)
    labels_ptr = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    lib.kmeans_clustering(pixel_ptr, num_pixels, num_centroids, max_iters, seed, centroids_ptr, labels_ptr)

    return centroids, labels

def reconstruct_image(centroids, labels, size):
    pixels = centroids[labels].astype(np.uint8)
    return Image.fromarray(pixels.reshape(size[1], size[0], 3), 'RGB')

def save_colormap(centroids, filename):
    with open(filename, 'w') as f:
        for color in centroids:
            f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image')
    parser.add_argument('output_py_image')
    parser.add_argument('output_c_image')
    parser.add_argument('colormap_py')
    parser.add_argument('colormap_c')
    parser.add_argument('--seed', type=int, default=214)
    parser.add_argument('--num_centroids', type=int, default=256)
    parser.add_argument('--max_iters', type=int, default=3)
    args = parser.parse_args()

    data, size, original_img = image_to_array(args.input_image)
    flat = flatten_image(data)

    print("Running Python KMeans...")
    start_py = time.time()
    py_centroids, py_labels = run_python_kmeans(flat, args.num_centroids, args.max_iters, args.seed)
    end_py = time.time()
    print(f"Python KMeans time: {end_py - start_py:.4f}s")

    py_image = reconstruct_image(py_centroids, py_labels, size)
    py_image.save(args.output_py_image)
    save_colormap(py_centroids, args.colormap_py)

    print("Running C KMeans...")
    start_c = time.time()
    c_centroids, c_labels = run_c_kmeans(flat, args.num_centroids, args.max_iters, args.seed)
    end_c = time.time()
    print(f"C KMeans time: {end_c - start_c:.4f}s")

    c_image = reconstruct_image(c_centroids, c_labels, size)
    c_image.save(args.output_c_image)
    save_colormap(c_centroids, args.colormap_c)

if __name__ == "__main__":
    main()
