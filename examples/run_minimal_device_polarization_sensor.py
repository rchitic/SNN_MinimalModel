##! /usr/bin/env python
# -*- coding: utf-8 -*-

from invertpy.sense.polarisation import MinimalDevicePolarisationSensor
from invertpy.io.sensor_params import save_eye2csv, load_csv2eye

import numpy as np
import sys


def main(*args):
    nb_lenses = 3
    omm_photoreceptor_angle = 1

    sensor = MinimalDevicePolarisationSensor(nb_lenses=nb_lenses, omm_photoreceptor_angle=omm_photoreceptor_angle)
    print(sensor)
    save_eye2csv(sensor, 'pol_compass')

    import matplotlib.pyplot as plt

    hue = sensor.hue_sensitive
    if hue.shape[0] == 1:
        print(f"Ommatidia xyz locations:\n {sensor.omm_xyz}")
        hue = np.vstack([hue] * sensor.omm_xyz.shape[0])
    rgb = hue[..., 1:4]
    rgb[:, [0, 2]] += hue[..., 4:5] / 2
    rgb[:, 0] += hue[..., 0]
    plt.subplot(111, polar=False)
    mask = sensor.omm_xyz[:, 2] > 0
    plt.scatter(sensor.omm_xyz[mask, 0],
                sensor.omm_xyz[mask, 1],
                s=20,
                c=np.clip(rgb[mask, :], 0, 1))
    # plt.ylim([-1.1, 1.1])
    # plt.xlim([-1.1, 1.1])
    plt.show()


if __name__ == '__main__':
    main(*sys.argv)
