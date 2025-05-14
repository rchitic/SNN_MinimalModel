"""
The Polarisation Sensor package that implements the sensor design and properties from [1]_.

References:
    .. [1] Gkanias, E., Risse, B., Mangan, M. & Webb, B. From skylight_gui input to behavioural output: a computational model
       of the insect polarised light compass. PLoS Comput Biol 15, e1007123 (2019).
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from invertpy.brain.compass import photoreceptor2pol, minimaldevice_photoreceptor2pol
from .vision import CompoundEye

from scipy.spatial.transform import Rotation as R

import numpy as np


class PolarisationSensor(CompoundEye):
    def __init__(self, nb_lenses=60, field_of_view=56, degrees=True, *args, **kwargs):
        """
        The bio-inspired polarised light sensor designed by [1]_.

        It takes as input the field of view and the number of lenses and creates dome the emulates the DRA of desert
        ants. It returns the responses of the POL neurons.

        Parameters
        ----------
        nb_lenses: int, optional
            the number of lenses for the sensor (equivalent to the `nb_inputs`)
        field_of_view: float, optional
            the field of view of the sensor is the angular diameter of the dome.
        degrees: bool, optional
            whether the field of view is given in degrees or not.

        Notes
        -----
        .. [1] Gkanias, E., Risse, B., Mangan, M. & Webb, B. From skylight_gui input to behavioural output: a computational
           model of the insect polarised light compass. PLoS Comput Biol 15, e1007123 (2019).

        """

        kwargs.setdefault('nb_input', nb_lenses)
        nb_inputs = kwargs['nb_input']
        if nb_inputs <= 8:
            nb_samples = [nb_inputs]
        elif nb_inputs <= 12:
            nb_samples = [4, nb_inputs - 4]
        elif nb_inputs <= 24:
            nb_samples = [4, 8, nb_inputs - 12]
        elif nb_inputs <= 40:
            nb_samples = [4, 8, 12, nb_inputs - 24]
        elif nb_inputs <= 60:
            nb_samples = [4, 8, 12, 16, nb_inputs - 40]
        else:
            nb_samples = None
        if nb_samples is not None:
            omm_sph = generate_rings(nb_samples=nb_samples, fov=field_of_view, degrees=degrees)[..., :2]
            omm_euler = np.hstack([omm_sph, np.full((omm_sph.shape[0], 1), np.pi / 2)])
            kwargs.setdefault('omm_ori', R.from_euler('ZYX', omm_euler, degrees=False))
        kwargs.setdefault('omm_rho', np.deg2rad(5.4))
        kwargs.setdefault('omm_pol_op', 1)
        kwargs.setdefault('c_sensitive', [0, 0, 1, 0, 0])
        kwargs.setdefault('name', 'pol_compass')
        kwargs.setdefault('nb_output', (nb_inputs,))
        super().__init__(*args, **kwargs)

        self._field_of_view = np.deg2rad(field_of_view)

    def _sense(self, sky=None, scene=None):
        r = super()._sense(sky=sky, scene=scene)

        # transform the photoreceptor signals to POL-neuron responses.
        return np.asarray(photoreceptor2pol(r, ori=self.omm_ori, nb_receptors=self._phot_angle,
                                            dtype=self.dtype).reshape((-1, 1)), dtype=self.dtype)

    def __repr__(self):
        return ("PolarisationSensor(ommatidia=%d, FOV=%.0f, responses=(%d, %d), pr_angles=%d, "
                "pos=(%.2f, %.2f, %.2f), ori=(%.2f, %.2f, %.2f), name='%s')") % (
            self.nb_ommatidia, np.rad2deg(self.field_of_view), self._nb_output[0], self._nb_output[1],
            len(self._phot_angle), self.x, self.y, self.z, self.yaw_deg, self.pitch_deg, self.roll_deg, self.name
        )

    @property
    def field_of_view(self):
        """
        The field of view of the sensor.
        """
        return self._field_of_view

    @property
    def nb_lenses(self):
        """
        The number of lenses of the sensor.
        """
        return self.nb_ommatidia


class MinimalDevicePolarisationSensor(PolarisationSensor):
    def __init__(self, POL_method="single_0", nb_lenses=3, omm_photoreceptor_angle=2, field_of_view=56, degrees=True, *args, **kwargs):
        kwargs.setdefault('name', 'minimal_device_pol_compass')
        super().__init__(nb_lenses, field_of_view, degrees, *args, **kwargs)
        self._phot_angle = self.process_omm_photoreceptor_angle(omm_photoreceptor_angle)
        self.POL_method = POL_method
        self.r_POL = np.zeros(nb_lenses)

    def _sense(self, sky=None, scene=None):
        """
        Transform the photoreceptor signals to POL-neuron responses.
        """
        r = super(PolarisationSensor, self)._sense(sky=sky, scene=scene)
        POL_responses = np.asarray(minimaldevice_photoreceptor2pol(r, POL_method=self.POL_method, ori=self.omm_ori, nb_receptors=self._phot_angle,
                                            dtype=self.dtype), dtype=self.dtype)
        if len(POL_responses) == 6:
            POL_responses_3 = np.zeros(3)
            POL_responses_3[0] = POL_responses[1] - POL_responses[4]
            POL_responses_3[1] = POL_responses[3] - POL_responses[0]
            POL_responses_3[2] = POL_responses[5] - POL_responses[2]
            POL_responses_3 = POL_responses_3 + 1 # make all responses positive (from range (-1,1) to (0,2))
            POL_responses_3 /= 2 # back to (0,1)
        else:
            POL_responses_3 = POL_responses
        self.r_POL = POL_responses_3
        return self.r_POL

def generate_rings(nb_samples, fov, degrees=True):
    """
    Generates concentric rings based on the number of samples parameter and the field of view, and places the lenses
    on the rings depending on the requested number of samples.

    Parameters
    ----------
    nb_samples: list | np.ndarray
        list containing the number of samples per ring.
    fov: float
        the angular diameter of the biggest ring.
    degrees: bool, optional
        whether the field of view is given in degrees or not.

    Returns
    -------
    samples: np.ndarray
        N x 3 array of the spherical coordinates (azimuth, elevation, distance) of the samples.
    """
    nb_rings = len(nb_samples)
    nb_samples_total = np.sum(nb_samples)
    if not degrees:
        fov = np.rad2deg(fov)
    v_angles = fov / float(2 * nb_rings + 1)

    phis = np.zeros(nb_samples_total, dtype='float32')
    thetas = np.zeros(nb_samples_total, dtype='float32')
    i = 0
    for r, samples in enumerate(nb_samples):
        theta = 90 + r * v_angles + v_angles / 2
        h_angles = 360. / samples
        for c in range(samples):
            if samples % 2 == 0:
                phi = c * h_angles
            else:
                phi = c * h_angles + h_angles / 2
            phis[i] = np.deg2rad(phi)
            thetas[i] = np.deg2rad(-theta)
            i += 1

    return np.vstack([phis, thetas, np.ones_like(phis)]).T

