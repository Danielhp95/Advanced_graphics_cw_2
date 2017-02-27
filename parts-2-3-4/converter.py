from __future__ import division
import numpy as np
from PNM import *

def gamma_correct(n, gamma):
    return n**(1/gamma)

sample_count = 1024

# load image
img = loadPFM("grace_latlong.pfm")

# compute intensity
intensity = img.sum(2) / 3

# pixels at the poles are more densely packed and should each have less chance of selection
intensity_sin = (intensity.transpose() * np.sin(np.linspace(0, np.pi, img.shape[0]))).transpose()

# compute pdfs based on intensity
# pdf_r is a single pdf to select the row
pdf_r = np.nan_to_num(intensity_sin.sum(1) / intensity_sin.sum())
# pdf_c contains a pdf for each row to select the pixel once the row is selected
pdf_c = np.nan_to_num(intensity_sin.transpose() / intensity_sin.sum(1)).transpose()
# convert to cdfs
cdf_r = pdf_r.cumsum()
cdf_c = pdf_c.cumsum(1)

q2_out = img.copy()

# get sample_count unique sample pixels based on cdfs
samples = set()
while len(samples) < sample_count:
    y = np.argmax(cdf_r > np.random.rand())
    x = np.argmax(cdf_c[y] > np.random.rand())
    samples.add((y, x))
    q2_out[max(0, y-2):y+3, max(0, x-2):x+3] = [0, 0, 1]

writePPM("out/q2_out_samples-{}.ppm".format(sample_count), (255*np.clip(gamma_correct(q2_out, 2.2), 0, 1)).astype(np.uint8))
writePFM("out/q2_out_samples-{}.pfm".format(sample_count), q2_out)

# ignore (this is just to visualise the probability of selecting each pixel)
q2_out_p = np.ndarray(img.shape)
p_px = (pdf_c.transpose()*pdf_r).transpose()
q2_out_p[:,:,0], q2_out_p[:,:,1], q2_out_p[:,:,2] = p_px, p_px, p_px
writePPM("out/q2_out_pixel_probability.ppm", (255*np.clip(gamma_correct(q2_out_p/q2_out_p.max(), 2.2), 0, 1)).astype(np.uint8))

# normalised luminosity of sampling each pixel
l = np.array([intensity[y, x] * img[y, x] / np.linalg.norm(img[y, x]) for y, x in samples])
# probability of sampling each pixel
p = np.array([pdf_r[y] * pdf_c[y, x] for y, x in samples])
# phi should also have -pi/2 added to it but for some reason it doesnt match part 4
phi = [x / img.shape[1] * 2 * np.pi for y, x in samples]
theta = [y / img.shape[0] * np.pi for y, x in samples]
# direction vector of each pixel in the Environment Map 
w = np.array([-np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)]).transpose()

# draw sphere using magic formula
rad = 255
diam = 2 * rad + 1
q3_out = np.zeros((diam, diam, 3))
for y, x in np.ndindex((diam, diam)):
    y_cen = 1 - y / rad
    x_cen = 1 - x / rad
    z_cen = np.sqrt(1 - x_cen**2 - y_cen**2)
    if not np.isnan(z_cen):
        q3_out[y, x] = np.mean((l.transpose() * np.clip(w.dot([x_cen, y_cen, z_cen]), 0, 1) / np.pi / p).transpose(), 0)
 
writePPM("out/q3_out_samples-{}.ppm".format(sample_count), (255*np.clip(gamma_correct(q3_out/q3_out.max(), 1.5), 0, 1)).astype(np.uint8))
writePFM("out/q3_out_samples-{}.pfm".format(sample_count), q3_out/q3_out.max())