from __future__ import division
import numpy as np
from PNM import *

def gamma_correct(n, gamma):
    return n**(1/gamma)

sample_count = 1024

img = loadPFM("grace_latlong.pfm")

intensity = img.sum(2) / 3

intensity_sin = (intensity.transpose() * np.sin(np.linspace(0, np.pi, img.shape[0]))).transpose()

pdf_r = np.nan_to_num(intensity_sin.sum(1) / intensity_sin.sum())
pdf_c = np.nan_to_num(intensity_sin.transpose() / intensity_sin.sum(1)).transpose()
cdf_r = pdf_r.cumsum()
cdf_c = pdf_c.cumsum(1)

out1 = img.copy()

samples = set()
while len(samples) < sample_count:
    y = np.argmax(cdf_r > np.random.rand())
    x = np.argmax(cdf_c[y] > np.random.rand())
    samples.add((y, x))
    out1[max(0, y-2):y+3, max(0, x-2):x+3] = [0, 0, 1]

writePPM("out1.ppm", (255*np.clip(gamma_correct(out1, 2.5), 0, 1)).astype(np.uint8))

l = np.array([intensity[y, x] * img[y, x] / np.linalg.norm(img[y, x]) for y, x in samples])
p = np.array([pdf_r[y] * pdf_c[y, x] for y, x in samples])
theta = [x / img.shape[1] * 2 * np.pi - np.pi / 2 for y, x in samples]
phi = [y / img.shape[0] * np.pi for y, x in samples]
w = np.array([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)]).transpose()

rad = 255
diam = 2 * rad + 1
out2 = np.zeros((diam, diam, 3))
for y, x in np.ndindex((diam, diam)):
    y_cen = 1 - y / rad
    x_cen = 1 - x / rad
    z_cen = np.sqrt(1 - x_cen**2 - y_cen**2)
    if not np.isnan(z_cen):
        out2[y, x] = np.mean((l.transpose() * np.clip(w.dot([x_cen, y_cen, z_cen]), 0, 1) / np.pi / p).transpose(), 0)

out2 /= out2.max()

writePPM("out2.ppm", (255*np.clip(out2, 0, 1)).astype(np.uint8))
writePFM("out2.pfm", out2)