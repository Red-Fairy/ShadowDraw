import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.fft import fft
import argparse
from PIL import Image
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from scipy import stats
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

def fractal_dimension(img):
    img = (img > 0)
    sizes = 2 ** np.arange(1, int(np.log2(min(img.shape))), 1)
    def box_count(img, k):
        S = np.add.reduceat(
            np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
                                np.arange(0, img.shape[1], k), axis=1)
        return np.count_nonzero(S)
    counts = [box_count(img, k) for k in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def compute_irregularity(input_dir):

    compactness_vals = []
    convexity_vals = []
    eccentricity_vals = []
    fractal_vals = []
    entropy_vals = []
    fourier_vals = []
    area_vals = []

    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')],
                        key=lambda x: float(os.path.splitext(x)[0].split('rot')[-1]))

    image_shape = np.array(Image.open(os.path.join(input_dir, image_files[0]))).shape[:2]

    border_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    border_mask[0, :] = 1
    border_mask[-1, :] = 1
    border_mask[:, 0] = 1
    border_mask[:, -1] = 1

    for filename in image_files:

        image = np.array(Image.open(os.path.join(input_dir, filename)))
        binary = cv2.Canny(image, 100, 200, apertureSize=7)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # post-process: for pixels that are 0 on the boundary in image, set the corresponding pixel in binary to 255
        condition = (image == 0) * (border_mask == 1)
        binary[condition] = 255

        labels = label(binary)
        props = regionprops(labels)
        if not props:
            print(f"No props for {filename}")
            continue
        largest = max(props, key=lambda p: p.area)
        mask = (labels == largest.label).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]

        area = np.sum(image == 0)
        peri = cv2.arcLength(cnt, closed=True)
        compactness = 4 * np.pi * area / (peri**2) if peri > 0 else 0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        convex_deficiency = (hull_area - area) / hull_area if hull_area > 0 else 0

        eccentricity = largest.eccentricity
        fractal_dim = fractal_dimension(255-image)

        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        radii = [np.sqrt((pt[0][0] - cx)**2 + (pt[0][1] - cy)**2) for pt in cnt]
        hist, _ = np.histogram(radii, bins=30, density=True)
        entropy = -np.sum(hist * np.log(hist + 1e-10))

        complex_coords = np.array([pt[0][0] + 1j * pt[0][1] for pt in cnt])
        fd = fft(complex_coords)
        high_freq_ratio = np.sum(np.abs(fd[30:])) / np.sum(np.abs(fd))

        compactness_vals.append(compactness)
        convexity_vals.append(convex_deficiency)
        eccentricity_vals.append(eccentricity)
        fractal_vals.append(fractal_dim)
        entropy_vals.append(entropy)
        fourier_vals.append(high_freq_ratio)
        area_vals.append(float(area))

    assert len(image_files) == len(compactness_vals) == len(convexity_vals) == len(eccentricity_vals) == len(fractal_vals) == len(entropy_vals) == len(fourier_vals)

    # save all values to a json file
    save_dict = {}
    for filename, area, compactness, convexity, eccentricity, fractal, entropy, fourier in zip(image_files, area_vals, compactness_vals, convexity_vals, eccentricity_vals, fractal_vals, entropy_vals, fourier_vals):
        save_dict[filename] = {
            'area': area,
            'compactness': compactness,
            'convexity': convexity,
            'eccentricity': eccentricity,
            'fractal': fractal,
            'entropy': entropy,
            'fourier': fourier,
        }

    return save_dict

class DistributionFitter:
    def __init__(self, x_data, y_data, degree=3):
        self.x_data = x_data
        self.y_data = y_data
        
        # Normalize y to be a proper probability
        self.y_normalized = y_data / np.trapezoid(y_data, x_data)
        
        # Fit polynomial to normalized data
        self.poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        self.poly_model.fit(x_data.reshape(-1, 1), self.y_normalized)
    
    def pdf(self, x):
        """Get probability density at x"""
        x = np.atleast_1d(x)
        y_pred = self.poly_model.predict(x.reshape(-1, 1))
        return np.maximum(y_pred, 1e-10)  # Ensure positive
    
    def sample(self, x_max=360, n_samples=1000):
        """Sample x values according to the fitted distribution"""
        x_range = np.linspace(0, x_max, 1000)
        pdf_values = self.pdf(x_range)
        
        # Normalize to sum to 1 for discrete sampling
        pdf_values = pdf_values / np.sum(pdf_values)
        
        # Sample using the PDF as weights
        samples = np.random.choice(x_range, size=n_samples, p=pdf_values)
        return samples

class InverseTransformSampler:
    def __init__(self, x_data, y_data):
        # Sort data by x
        sorted_indices = np.argsort(x_data)
        self.x_sorted = x_data[sorted_indices]
        self.y_sorted = y_data[sorted_indices]
        
        # Normalize to probability
        self.y_normalized = self.y_sorted / np.trapezoid(self.y_sorted, self.x_sorted)
        
        # Compute CDF
        self.cdf_values = np.concatenate([[0], cumulative_trapezoid(self.y_normalized, self.x_sorted)])
        self.cdf_values = self.cdf_values / self.cdf_values[-1]  # Ensure it ends at 1
        
        # Create inverse CDF function
        self.inverse_cdf = interp1d(self.cdf_values, self.x_sorted, 
                                   bounds_error=False, fill_value=(0, 100))
    
    def sample(self, n_samples=1000):
        """Sample using inverse transform method"""
        u = np.random.uniform(0, 1, n_samples)
        return self.inverse_cdf(u)
    
    def pdf(self, x):
        """Approximate PDF using interpolation"""
        pdf_interp = interp1d(self.x_sorted, self.y_normalized, 
                             bounds_error=False, fill_value=0)
        return pdf_interp(x)

if __name__ == '__main__':

    json_data_path = '/home/rl897/art-from-phys/gradio_demo/results/real_object_kushal/renderings/right_shoe/irregularity_with_object_hull.json'
    with open(json_data_path, 'r') as f:
        data = json.load(f)

    area_exp_factor = 0.25
    rank_factor = 25.0

    x_data, y_data = [], []
    for image_name, irregularity_data in data.items():
        x_this, y_this = float(image_name.split('rot')[-1][:6]), np.exp(((irregularity_data['convexity'] - irregularity_data['compactness']) + ((irregularity_data['area'] / 1e4) ** area_exp_factor)) * rank_factor)
        x_data.append(x_this)
        y_data.append(y_this)
        if x_this == 0:
            x_data.append(360)
            y_data.append(y_this)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Fit the distribution
    # fitter = DistributionFitter(x_data, y_data)
    fitter = InverseTransformSampler(x_data, y_data)

    # Sample new x values
    samples = fitter.sample(1000)

    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(x_data, y_data, alpha=0.7, label='Original data')
    x_smooth = np.linspace(0, 360, 200)
    y_smooth = fitter.pdf(x_smooth) * np.trapezoid(y_data, x_data)  # Scale back up
    plt.plot(x_smooth, y_smooth, 'r-', label='Fitted distribution')
    plt.xlabel('x')
    plt.ylabel('y (score)')
    plt.legend()
    plt.title('Original Data and Fitted Distribution')

    plt.subplot(1, 2, 2)
    plt.hist(samples, bins=30, density=True, alpha=0.7, label='Samples')
    plt.xlabel('x')
    plt.ylabel('Probability density')
    plt.title('Sampled x values')
    plt.legend()

    plt.tight_layout()
    plt.savefig('distribution_fitting.png')

    plt.close()