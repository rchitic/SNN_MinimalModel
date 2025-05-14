
__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.1-alpha"
__maintainer__ = "Evripidis Gkanias"

import warnings

from .centralcomplex import CentralComplexBase
from .ellipsoidbody import SimpleCompass, PontineSteering, MinimalDeviceSteering
from .fanshapedbody import PathIntegratorLayer, MinimalDevicePathIntegratorLayer
from .fanshapedbody_dye import MinimalDevicePathIntegrationDyeLayer
from ._helpers import tn_axes
from invertpy.sense.polarisation import MinimalDevicePolarisationSensor
from invertpy.brain.synapses import chessboard_synapses

class MinimalDeviceCX(CentralComplexBase):

    def __init__(self, POL_method="single_0", omm_photoreceptor_angle=2, field_of_view=56, degrees=True, nb_direction=3, nb_memory=3, tau=135518, b_c=1.164, update=True, use_nanowires=False, sigmoid_bool=True, use_dye=False, nb_sigmoid=6, nb_steer=2, a=0.667, b_s=4.372, *args, **kwargs):
        """
        The Central Complex model of [1]_ as a component of the locust brain.

        Parameters
        ----------
        nb_direction: int, optional
            the number of direction neurons.
        nb_memory: int, optional
            the number of memory neurons.
        nb_sigmoid: int, optional
            the number of sigmoid neurons
        nb_steer: int, optional
            the number of steering neurons.

        Notes
        -----
        .. [1] Stone, T. et al. An Anatomically Constrained Model for Path Integration in the Bee Brain.
           Curr Biol 27, 3069-3085.e11 (2017).
        """

        super().__init__(*args, **kwargs)
        if not use_nanowires:
            self["compass"] = MinimalDevicePolarisationSensor(POL_method, nb_lenses=nb_direction, omm_photoreceptor_angle=omm_photoreceptor_angle, field_of_view=field_of_view, degrees=degrees, *args, **kwargs)

        if use_dye:
            self["memory"] = MinimalDevicePathIntegrationDyeLayer(nb_direction=nb_direction, nb_memory=nb_memory, tau=tau, b_c=b_c, update=update, sigmoid_bool=sigmoid_bool)
        else:
            self["memory"] = MinimalDevicePathIntegratorLayer(nb_direction=nb_direction, nb_memory=nb_memory, tau=tau, b_c=b_c, update=update, sigmoid_bool=sigmoid_bool)

        self["steering"] = MinimalDeviceSteering(nb_direction=nb_direction, nb_memory=nb_memory, nb_sigmoid=nb_sigmoid, nb_steer=nb_steer, a=a, b_s=b_s)

        self.nb_direction = nb_direction
        self.nb_memory = nb_memory
        self.nb_sigmoid = nb_sigmoid
        self.nb_steer = nb_steer

        w = chessboard_synapses(self.nb_steer, 2, nb_rows=2, nb_cols=2, fill_value=1, dtype=self.dtype)
        self._w_s2o = w

        if self.__class__ == MinimalDeviceCX:
            self.reset()

    def __repr__(self):
        return f"MinimalDeviceCX(direction={self.nb_direction}, memory={self.nb_memory}, " \
               f"sigmoid={self.nb_sigmoid}, steering={self.nb_steer})"

    def _fprop(self, POL_direction):
        memory = self.memory(direction=POL_direction)
        steering = self.steering(direction=POL_direction, memory=memory)
        return steering

    def reset_integrator(self):
        if hasattr(self.memory, 'reset_integrator'):
            self.memory.reset_integrator()
        else:
            warnings.warn("There is no integrator to reset.")

    @property
    def steering2motor(self):
        """
        Matrix transforming the steering responses to their contribution to the motor commands.
        """
        return self._w_s2o

    @property
    def w_steering2motor(self):
        return self._w_s2o

    @property
    def steering(self):
        """

        Returns
        -------
        MinimalDeviceSteering
        """
        return self["steering"]

    @property
    def compass(self):
        """

        Returns
        -------
        MinimalDevicePolarisationSensorCompass
        """
        return self["compass"]

    @property
    def memory(self):
        """

        Returns
        -------
        MinimalDevicePathIntegratorLayer
        """
        return self["memory"]

    @property
    def r_direction(self):
        return self.compass.r_POL

    @property
    def r_memory(self):
        return self.memory.r_memory

    @property
    def r_sigmoid_neuron(self):
        return self.steering.r_sigmoid_neuron

    @property
    def r_steering(self):
        return self.steering.r_steering

    @property
    def r_steering_diff(self):
        return self.steering.r_steering_diff

    @property
    def r_motor(self):
        return self.steering.r_steering.dot(self.w_steering2motor)

