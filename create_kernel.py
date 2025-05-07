import cv2
import numpy as np
from PIL import Image
import os

output_dir = 'datasets/text/kernel'
os.makedirs(output_dir, exist_ok=True)

num_kernels = len(os.listdir('datasets/text/train'))  # same as number of AB images

for i in range(num_kernels):
    size = np.random.choice([11, 15, 21])      # kernel size
    sigma = np.random.uniform(0.5, 2.0)        # blur strength

    # Create Gaussian kernel
    k1d = cv2.getGaussianKernel(size, sigma)
    kernel = k1d @ k1d.T                      # 2D Gaussian kernel

    kernel = kernel / kernel.sum()           # normalize
    kernel_img = (kernel * 255).astype(np.uint8)
    im = Image.fromarray(kernel_img)

    kernel_path = os.path.join(output_dir, f'{i:04d}.png')
    im.save(kernel_path)
