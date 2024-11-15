""" Class definition and functions related to RF emulation functionality:
- RFFrontend: Definition of RF frontend, holds multiple antennas
    - Antenna: Definition of antenna object, holds multiple beams and local properties (Beampattern, 
        NoiseFigure, Temperatures, etc.)
        - Beam: Definition of beam object, holds beam pattern, beam orientation (object), and local
        properties (Transmit power and bandwidth, visualization settings, etc.)
        - Beamorientation: Definition of beam orientation object, provides special functionality for
        conveniently controling beam orientation with all degrees of freedom.
- BeamPattern: Definition of a beam pattern interface and predefined beam patterns (Uniform, TR38.901, Element Array).

- RFSource: Object for convenient RF IQ data generation in a background thread based on a user-defined generator function.
- RFSink: Object for convenient RF IQ data reception in a background thread based on a user-defined receiver function.

- RFEmulator: Definition of RF Emulator object, managing and orchestrating efficient processing of registered
    LinkEnvironments for handling singular and multi-component links.
    - LinkEnvironmentGroundToSpace / LinkEnvironmentGroundToSpaceGroup: Definition of a link environment object, 
        holding high-level link properties
        - Link: Definition of a link object, holding low-level link properties and implementing efficient IQ emulation

- LinkLevelGroundToSpace: Definition of a convenience object for computing various 3GPP-compliant link-related properties.

Todo @Michael: Complete function documentation and missing typing hints

Author: Michael Petry
Last Edit: 27.10.2024

Documentation is in Google Docstrings Format.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Union, Callable
if TYPE_CHECKING: # enables type checking while avoiding circular import in runtime
    from pylib.Propagatables import Propagatable
    from pylib.Stationary import Stationary
import threading
import numpy as np
from pylib.Utils import BlockingCircularBuffer, ModCod_LUT, RealTiming, randUUID, beamPatterntoLUT, beamPatternToModel3D, pdist2, getSlantRange, getPyModule, spicePosTo3JSPos
from pylib.InheritanceUtils import *
from pylib.SceneComponent import SceneComponent, WithOrientation
from pylib.Messages import Message
import asyncio
import threading
from websockets.server import serve
import logging
from itertools import chain
from enum import Enum
import struct
import vector

##############################################################
### Enable experimental GPU acceleration for IQ processing ###
### - via Tensorflow as Proxy                              ###
### - via OpenCL as custom kernel                          ###
##############################################################
RFEMULATOR_GPU_IQ_OFFLOAD_TENSORFLOW = False 
RFEMULATOR_GPU_IQ_OFFLOAD_OPENCL = False # not currently embedded in this branch

class RFFrontend(RequireEncode):
        r"""Encodable object holding information about RF Frontend definition. Holds a list of antennas.

        Args:
            antennas (List['Antenna']): List of antennas in the RF frontend.

        Notes:
        
        The 'Antenna' objects in the provided list will be modified on object instanciation to store the 
        RF frontend object as parent and are assigned an antenna id. The nested 'Beam' objects in each 
        'Antenna' object will be modified to store the parent antenna id.
        """

   
        def __init__(self, antennas: List['Antenna']):
            for i in range(len(antennas)): # set antenna idx to index in list for ease of use internally.
                antennas[i]._antIdx = i        
                antennas[i]._rfFrontend = self        
                for b in antennas[i].getBeams():
                    b._antIdx = i # store antenna idx in beams for ease of use internally.
            self._antennas = antennas

        def getAntennas(self,) -> List['Antenna']:
            r"""Returns the list of registered antennas in the RF frontend.
            
            Returns:
                List['Antenna']: List of antennas registered in the RF frontend.
            """
            return self._antennas

        def getCertainAntenna(self,ant_idx) -> 'Antenna':
            r"""Returns the antenna object with the specified index, which is the same as the antenna id.
            
            Args:
                ant_idx: Index of the antenna object to return.

            Returns:
                'Antenna': Antenna object with the specified index
            """
            return self._antennas[ant_idx]
        
        def _setOwner(self, owner: 'SceneComponent'):
            self._owner = owner
            for i in range(len(self._antennas)): # set antenna idx to index in list for ease of use internally. 
                for b in self._antennas[i].getBeams():
                    b.orientation()._setOwner(owner = owner, hasParentOrientation=True)

        def getOwner(self,) -> 'SceneComponent':
            r"""Returns the parent object of the RF frontend.
            
            Returns:
                'SceneComponent': Parent object of the RF frontend.
            """
            return self._owner
        
        def _updateOrientation(self,):
            for i in range(len(self._antennas)): # set antenna idx to index in list for ease of use internally. 
                for b in self._antennas[i].getBeams():
                    b.orientation().update()

        
        def encode(self, messagetype: Message.Type) -> Dict:
            # Internal method for encoding the RF frontend object.
            assert self.getOwner() != None, "RFFrontend's owner not set."
            return {
                "object_type": self.__class__.__name__,
                "antennas": self._antennas,
            }
        
        class Antenna(RequireEncode):
            r"""Encodable object holding information about the antenna definition. Holds a list of beams.
            
            Args:
                beams (List['Beam']): List of beams in the antenna.
                noiseFigure: Optional[float]: Noise figure of the antenna in dB.
                antennaTemperature: Optional[float]: Antenna temperature in Kelvin.
                ambientTemperature: Optional[float]: Ambient temperature in Kelvin.

            Notes:

            The beam objects in the provided list will be modified on object instanciation and assigned the
            beam id (local to the antenna object) and the antenna object as parent.

            """
            
            def __init__(self, beams: List['Beam'], noiseFigure: Optional[float]=6, antennaTemperature: Optional[float]=280, ambientTemperature: Optional[float]=280):
                for i in range(len(beams)): # set beams idx to index in list for ease of use internally.
                    beams[i]._beamIdx = i
                    beams[i]._antenna = self
                self._beams = beams
                self._noiseFigure = noiseFigure
                self._antennaTemperature = antennaTemperature
                self._ambientTemperature = ambientTemperature                

            def getBeams(self,) -> List['Beam']:
                r"""Returns the list of registered beams in the antenna.
                
                Results:
                    List['Beam']: List of beams registered in the antenna.
                """
                return self._beams

            def getCertainBeam(self,beam_idx) -> 'Beam':
                r"""Returns the beam object with the specified index, which is the same as the beam id.
                
                Args:
                    beam_idx: Index of the beam object to return.
                
                Returns:
                    'Beam': Beam object with the specified index.
                """
                return self._beams[beam_idx]
            
            def getNoiseFigure(self) -> float:
                r"""Returns the noise figure of the antenna in dB.
                
                Returns:
                    float: Noise figure of the antenna in dB.
                """
                return self._noiseFigure

            def getAntennaTemperature(self,) -> float:
                r"""Returns the antenna temperature in Kelvin.
                
                Returns:
                    float: Antenna temperature in Kelvin.
                """
                return self._antennaTemperature
            
            def getAmbientTemperature(self,) -> float:
                r"""Returns the ambient temperature in Kelvin.
                
                Returns:
                    float: Ambient temperature in Kelvin.
                """
                return self._ambientTemperature
            
            def getRFFrontend(self,) -> 'RFFrontend':
                r"""Returns the parent RF frontend object.
                
                Returns:
                    'RFFrontend': Parent RF frontend object of the antenna.
                """
                return self._rfFrontend
            
            def encode(self, messagetype: Message.Type) -> Dict:
                # Internal method for encoding the antenna object.
                return {
                    "object_type": self.__class__.__name__,
                    "beams": self._beams,
                }
            

            class BeamOrientation(WithOrientation.Orientation):
            #class BeamOrientation(SatelliteOrientation):
                r"""Object holding information about the beam orientation. Provides special functionality for
                conveniently controling the beam orientation with all degrees of freedom relative to spacecraft
                velocity, position, and target (including tracking).

                Args:
                    owner (SceneComponent): Parent object (e.g., Beam) of the beam orientation object.
                    hasParentOrientation (Optional[bool]): Flag indicating if the objects behaviour should be
                        inferred from the owner object's parent object's Beamorientation object, if it exists. Default is False.
                        If True, this allows to simply control the beam's orientation properties by setting the satellite
                        orientation properties, e.g., satellite target = beam target.
                    
                Notes:
                
                The 'update' method handles functionality of the convenience functions and is called automatically
                by the scenario runtime management every simulation timestep. It must not be called manually.
                
                """
        
                class Setting(Enum):
                    ZUP = 0
                    ZDOWN = 1
                    YUP = 2
                    YDOWN = 3

                def __init__(self, owner: 'Satellite', hasParentOrientation: Optional[bool]=False):
                    super().__init__(owner=owner, hasParentOrientation=hasParentOrientation)
                    self._setting = None
                    self._lookAtTrack = False
                    self._lookAtTrackObj = None

                def lookAt(self, vec3):
                    r"""Points object's x axis to a target specified as cartesian 3D position.
                    
                    Args:
                        vec3 (List[float]): Target position as [x, y, z].
                    """
                    self.set_manual_orientation_lookat(vec3)
                    self._lookAtTrack = False
                    self._lookAtTrackObj = None

                def lookAtObject(self, obj, track: Optional[bool]=True):
                    r"""Points object's x axis to a target specified as SceneComponent.
                    Optionally tracks the target object's position (enabled by default).
                    
                    Args:
                        obj (SceneComponent): Target object.
                        track (Optional[bool]): Flag indicating if the beam orientation should track the target object.
                    
                    """
                    self._lookAtTrack = track
                    objPos = obj.getPosition()
                    if track:
                        self._lookAtTrackObj = obj
                    else:
                        self._lookAtTrackObj = None
                    if objPos != None:
                        self.set_manual_orientation_lookat(objPos)
                    else:
                        self._lookAtTrackObj = obj # setting object despite track=False means update it *once* in update period, because position was not yet valid.


                def setZUp(self,):
                    r"""Sets the beam orientation to point the z-axis upwards."""
                    self._setting = RFFrontend.Antenna.BeamOrientation.Setting.ZUP
                    self.set_manual_orientation_up([0,0,1])

                def setZDown(self,):
                    r"""Sets the beam orientation to point the z-axis downwards."""
                    self._setting = RFFrontend.Antenna.BeamOrientation.Setting.ZDOWN
                    self.set_manual_orientation_up([0,0,-1])

                def setYUp(self,):
                    r"""Sets the beam orientation to point the y-axis upwards."""
                    self._setting = RFFrontend.Antenna.BeamOrientation.Setting.YUP
                    # up set in update loop

                def setYDown(self,):
                    r"""Sets the beam orientation to point the y-axis downwards."""
                    self._setting = RFFrontend.Antenna.BeamOrientation.Setting.YDOWN
                    # up set in update loop

                def update(self,):
                    # Internal method to continuously update the beam orientation (e.g., when tracking).
                    if self._lookAtTrackObj != None:
                        self.set_manual_orientation_lookat(self._lookAtTrackObj.getPosition())
                        if not self._lookAtTrack:
                            self._lookAtTrackObj = None # keep only tracking if track is true.

                    if self._setting == RFFrontend.Antenna.BeamOrientation.Setting.ZUP or self._setting == RFFrontend.Antenna.BeamOrientation.Setting.ZDOWN:
                        pass
                        # do nothing. this is the only that works without continuous update
                    elif self._setting == RFFrontend.Antenna.BeamOrientation.Setting.YUP or self._setting == RFFrontend.Antenna.BeamOrientation.Setting.YDOWN:
                        curpos = self._getOwner().getPosition()
                        # todo update michael: use dir to target to make it also work for non zero lookat pos, not absolute position.
                        target_vec = vector.obj(x=self._get_orientation_lookat()[0], y=self._get_orientation_lookat()[1], z=self._get_orientation_lookat()[2])
                        curpos_vec = vector.obj(x=curpos[0], y=curpos[1], z=curpos[2])
                        dir_vec = target_vec.subtract(curpos_vec)
                        dirrot_vec = dir_vec.rotate_axis(vector.obj(x=0, y=0, z=1), angle=-0.01)
                        up = dirrot_vec.subtract(dir_vec).unit()
                        if self._setting == RFFrontend.Antenna.BeamOrientation.Setting.YDOWN:
                            up = up.scale(factor=-1.0)
                        self.set_manual_orientation_up([up.x, up.y, up.z])

            class Beam(RequireEncode, WithOrientation[BeamOrientation]):
                r"""Encodable object holding information about the beam definition. Holds a beam pattern and
                corresponding orientation, transmit power and bandwidth, and visualization settings.

                Args:
                    pattern (BeamPattern): Beam pattern object.
                    transmitPower (Optional[float]): Transmit power in dBm.
                    transmitBandwidth (Optional[float]): Transmit bandwidth in Hz.
                    visualize3d (Optional[bool]): Flag indicating if the beam pattern should be visualized as 3D shape.
                    project2d (Optional[bool]): Flag indicating if the beam pattern should be projected on 2D planet surface.
                    scale (Optional[float]): Scale factor for the 3D visualization.
                    resolution_2d (Optional[int]): Resolution for the 2D projection.
                    resolution_3d (Optional[int]): Resolution for the 3D visualization.
                    project2dfactor (Optional[float]): Factor for defining the weight (strength) of this beam's 2D projection
                        when in superposition with other overlapping 2D projections.

                Notes:

                The static class variables 'list_encode_hashes_model3d' and 'list_encode_hashes_project2d' are used to
                optimize memory usage and communication bandwidth (when serializing and transmitting the beam pattern
                and generated 3D model to the frontend) by storing only unique versions of beam patterns 
                (e.g., when a satellite/antenna has capabilities for generating multiple beams of the same type/pattern). 
                This feature works globally for any Beam mesh and pattern in the scenario, determined by the beam patterns hash value.

                """

                list_encode_hashes_model3d = [] # static class variable, shared across all Beam instances.
                list_encode_hashes_project2d = [] # static class variable, shared across all Beam instances.

                def __init__(self, pattern: 'BeamPattern', transmitPower: Optional[float]=None, transmitBandwidth: Optional[float]=None, visualize3d: Optional[bool]=True, project2d: Optional[bool]=True, scale: Optional[float]=1, resolution_2d: Optional[int]=100, resolution_3d: Optional[int]=100, project2dfactor: Optional[float]=0.1):
                    WithOrientation.__init__(self, orientation_cls = RFFrontend.Antenna.BeamOrientation)
                    self._uuid = randUUID()
                    self._pattern = pattern
                    self._transmitPower = transmitPower # todo, put this in link environment
                    self._transmitBandwidth = transmitBandwidth # todo, put this in link environment
                    self._visualize3d = visualize3d
                    self._project2d = project2d
                    if visualize3d:
                        [model3d, hash_] = beamPatternToModel3D(pattern, resolution_3d)
                        self._model3d = [model3d[1], model3d[2], model3d[0], model3d[3]]
                        self._model3d_hash = hash_
                        self._model3d_scale = scale
                    if project2d:
                        [pattern, hash_] = beamPatterntoLUT(pattern, resolution_2d, project2dfactor)
                        self._project2d_pattern = pattern
                        self._project2d_pattern_hash = hash_
                
                def getAntenna(self,) -> 'RFFrontend.Antenna':
                    r"""Returns the parent antenna object.
                    
                    Returns:
                        'RFFrontend.Antenna': Parent antenna object.
                    """
                    return self._antenna
                
                def setProject2dVisible(self, visible: bool):
                    r"""Enables or disables the 2D projection visualization of the beam pattern.
                    
                    Args:
                        bool: Visibility state of the 2D projection.
                    """
                    self._project2d = visible                

                def calculateGain(self, target: List[float]):
                    r"""Calculates the gain of the beam pattern with its defined orientation for a given target position.
                    
                    Args:
                        target (List[float]): Target position as [x, y, z].

                    Returns:
                        float: Gain of the beam pattern in the direction of the target position.
                    """
                    def normalize(v):
                        norm = np.linalg.norm(v)
                        if norm == 0: 
                            return v
                        return v / norm

                    def cross(a, b):
                        return np.cross(a, b)

                    def dot(a, b):
                        return np.dot(a, b)

                    target = np.array(target, dtype=np.float32)
                    beamLookAt = np.array(self.orientation()._get_orientation_lookat(), dtype=np.float32)
                    beamPos = np.array(self.getAntenna().getRFFrontend().getOwner().getPosition(), dtype=np.float32)
                    beamUp = np.array(self.orientation()._get_orientation_up(), dtype=np.float32)
                    direction = normalize(target - beamPos)

                    # Calculate forward, up, and right vectors
                    forward = normalize(beamLookAt - beamPos)
                    up = normalize(beamUp)
                    right = normalize(cross(forward, up))
                    up = normalize(cross(forward, right))

                    # Transform the direction vector into the satellite's local coordinate system
                    localDirection = np.array([
                        dot(direction, right),
                        dot(direction, up),
                        dot(direction, forward)
                    ])

                    # Convert local direction to azimuth (phi) and elevation (theta) angles
                    phi = np.arctan2(localDirection[0], -localDirection[2])
                    theta = np.arccos(localDirection[1])
                    return self._pattern.calculateGain_Linear(theta=theta, phi=phi)
                
                def encode(self, messagetype: Message.Type) -> Dict:
                    # Internal method for encoding the beam object.
                    if messagetype == Message.Type.SCENARIOSETUP:
                        obj = {"object_type": self.__class__.__name__,                               
                            "orientation_lookat": spicePosTo3JSPos(self.orientation()._get_orientation_lookat()),                       
                            "orientation_up": spicePosTo3JSPos(self.orientation()._get_orientation_up()),
                            "orientation_showaxes": self.orientation()._objectAxesSize if self.orientation()._showObjectAxes else None,
                            }
                                              
                        if self._visualize3d:
                            if self._model3d_hash in self.list_encode_hashes_model3d: #encode same beam patterns only once, send hash instead (see Note)
                                obj["model3d_hash"] = self._model3d_hash                         
                            else:
                                obj["model3d"] = self._model3d
                                obj["model3d_hash"] = self._model3d_hash
                                self.list_encode_hashes_model3d.append(self._model3d_hash)
                            obj["model3d_scale"] = self._model3d_scale

                        if self._project2d:
                            if self._project2d_pattern_hash in self.list_encode_hashes_project2d:
                                obj["project2d_pattern_hash"] = self._project2d_pattern_hash
                            else:
                                obj["project2d_pattern"] = self._project2d_pattern
                                obj["project2d_pattern_hash"] = self._project2d_pattern_hash
                                self.list_encode_hashes_project2d.append(self._project2d_pattern_hash)

                        return obj
                           
                    elif messagetype == Message.Type.PROPAGATIONDATA:
                        return {
                            "object_type": self.__class__.__name__,                         
                            "orientation_lookat": spicePosTo3JSPos(self.orientation()._get_orientation_lookat()),                       
                            "orientation_up": spicePosTo3JSPos(self.orientation()._get_orientation_up()),
                            **({"project2d_pattern_hash": self._project2d_pattern_hash if self._project2d else "-1"} if hasattr(self, "_project2d_pattern_hash") else {}), # make projection visibility updatable
                        }

                def getTransmitPower(self,) -> float:
                    r"""Returns the transmit power in dBm.
                    
                    Returns:
                        float: Transmit power associated with the beam in dBm.
                    """
                    # Todo, move to Link
                    return self._transmitPower
                def getTransmitBandwidth(self,) -> int:
                    r"""Returns the transmit bandwidth in Hz.
                    
                    Returns:
                        int: Transmit bandwidth associated with the beam in Hz.
                    """
                    # Todo, move to Link
                    return self._transmitBandwidth
                    
                    
class BeamPattern(ABC):
    r"""Abstract base class for beam pattern objects. Provides a abstract function for calculating the gain 
    (linear scale) of a beam pattern. Also provides static methods for loading predefined beam patterns.
    
    Notes:

    A 3D mesh, 2D projection, and hash of the beam pattern will be automatically generated by the 'beamPatternToModel3D'
    and 'beamPatterntoLUT' utility function in Utils.py.
    
    """

    def __init__(self,):
        pass

    @abstractmethod
    def calculateGain_Linear(self, theta: float, phi: float, frequency: Optional[float]=None) -> float:
        r"""Abstract method for calculating the (linear scale) gain of the beam pattern at a given azimuth and elevation
        angle and frequency. For example implementations see the 'Uniform', 'TR38901', and 'ElementArray'
        implementation.

        Args:
            theta (float): Elevation angle in radians.
            phi (float): Azimuth angle in radians.
            frequency (Optional[float]): Frequency in Hz. Default is None.

        Returns:
            float: Linear scale gain of the beam pattern.
        """
        pass

    @staticmethod
    def Uniform(gain_linear: Optional[float]=1.0):
        return _Uniform(gain_linear=gain_linear)

    @staticmethod
    def customAntenna(pattern):
        return _customAntenna(pattern)
    
    @staticmethod
    def TR38901(cutoff: Optional[bool]=False):
        return _TR38901(cutoff=cutoff)
    
    @staticmethod
    def fromElementArray(num_col_el, num_row_el, element_spacing, az_range, el_range, center_frequency, stepsize):
        return _ElementArray(num_col_el=num_col_el,
                             num_row_el=num_row_el,
                             element_spacing=element_spacing,
                             az_range=az_range,
                             el_range=el_range,
                             center_frequency=center_frequency,
                             stepsize=stepsize)

class _Uniform(BeamPattern):
    r"""Implementation of a uniform beam pattern. The gain is constant across all azimuth and elevation angles."""

    def __init__(self, gain_linear):
        self._gain_linear = gain_linear
    def calculateGain_Linear(self, theta: float, phi: float):
        if isinstance (theta, np.ndarray):
            return np.ones(theta.shape, dtype=np.float32)*self._gain_linear # preserve input shape in output shape
        else:
            return self._gain_linear

class _customAntenna(BeamPattern):
    def __init__(self, pattern):
        self._pattern = pattern

    def calculateGain_Linear(self,theta: float, phi: float):
        if isinstance(theta, np.ndarray):
            from scipy.interpolate import griddata
            elevation = np.linspace(0, np.pi, self._pattern.shape[1]) # Assuming azimuth from 0 to pi
            azimuth = np.linspace(-np.pi, np.pi, self._pattern.shape[0]) # Assuming elevation from -pi to pi
            Azimuth, Elevation = np.meshgrid(azimuth, elevation)

            # Flatten the meshgrid and antenna gain matrix for interpolation
            points = np.vstack((Azimuth.flatten(), Elevation.flatten())).T
            values = self._pattern.flatten()

            new_points = np.vstack((phi.flatten(), theta.flatten())).T
        
            # Interpolate the values
            interpolated_values = griddata(points, values, new_points, method='linear')

            # Reshape the interpolated values to the new meshgrid shape
            g = interpolated_values.reshape(phi.shape)
            g = np.nan_to_num(g,nan=0.0)
            g[g<0.0] = 0.0
            self._pattern = g
            return g
        else:
            theta_idx = round(theta/180)
            phi_idx = round(phi/360)
            return self._pattern[phi_idx,theta_idx]


class _TR38901(BeamPattern):   
    r"""Implementation of a TR38901 beam pattern. The gain is calculated based on the 3GPP TR38.901 standard.
    
    Note:

    An optional cutoff parameter can be set to limit the beam pattern to the upper hemisphere for debugging purposes
    (e.g., orientation verification for a symmetric beam pattern).

    Credits:

    This implementation is extracted from Sionna and ported to numpy by Michael.
    (https://nvlabs.github.io/sionna/_modules/sionna/rt/antenna.html)

    """ 
    def __init__(self, cutoff: Optional[bool]=False):
        self._cutoff = cutoff

    def calculateGain_Linear(self, theta: float, phi: float):
        def polarization_model_1(c, theta, phi, slant_angle):
            # Example implementation for polarization model 1
            cos_slant = np.cos(slant_angle)
            sin_slant = np.sin(slant_angle)
            c_theta = c * cos_slant
            c_phi = c * sin_slant
            return c_theta, c_phi

        def polarization_model_2(c, slant_angle):
            # Example implementation for polarization model 2
            cos_slant = np.cos(slant_angle)
            sin_slant = np.sin(slant_angle)
            c_theta = c * cos_slant
            c_phi = c * sin_slant
            return c_theta, c_phi

        def tr38901_pattern(theta, phi, slant_angle=0.0, polarization_model=2, dtype=np.complex64):
            """
            Antenna pattern from 3GPP TR 38.901 (Table 7.3-1)

            Args:
                theta: array_like, float
                    Zenith angles wrapped within [0,pi] [rad]

                phi: array_like, float
                    Azimuth angles wrapped within [-pi, pi) [rad]

                slant_angle: float
                    Slant angle of the linear polarization [rad].
                    A slant angle of zero means vertical polarization.

                polarization_model: int, one of [1,2]
                    Polarization model to be used. Options `1` and `2` refer to
                    `polarization_model_1` and `polarization_model_2`, respectively.
                    Defaults to `2`.

                dtype : np.complex64 or np.complex128
                    Datatype.
                    Defaults to `np.complex64`.

            Output:
                g (np.ndarray): Flat array of linear gain factors for this antenna pattern.

            """
            rdtype = np.float32 if dtype == np.complex64 else np.float64
            theta = np.asarray(theta, dtype=rdtype)
            phi = np.asarray(phi, dtype=rdtype)
            slant_angle = np.asarray(slant_angle, dtype=rdtype)

            # Wrap phi to [-PI, PI]
            phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi

            if theta.shape != phi.shape:
                raise ValueError("theta and phi must have the same shape.")
            if polarization_model not in [1, 2]:
                raise ValueError("polarization_model must be 1 or 2")

            theta_3db = phi_3db = np.asarray(65 / 180 * np.pi, dtype=rdtype)
            a_max = sla_v = 30
            g_e_max = 8

            a_v = -np.minimum(12 * ((theta - np.pi / 2) / theta_3db) ** 2, sla_v)
            a_h = -np.minimum(12 * (phi / phi_3db) ** 2, a_max)
            a_db = -np.minimum(-(a_v + a_h), a_max) + g_e_max
            a = 10 ** (a_db / 10)
            c = np.array(np.sqrt(a), dtype=dtype)
            if self._cutoff:                
                c = np.where((theta > np.pi/2) & (phi > 0), c, 0)

            if polarization_model == 1:
                return polarization_model_1(c, theta, phi, slant_angle)
            else:
                return polarization_model_2(c, slant_angle)

        c_theta, c_phi = tr38901_pattern(theta, phi)        
        g = np.abs(c_theta)**2 + np.abs(c_phi)**2
        return g

class _ElementArray(BeamPattern):
    r"""Implementation of a rectangular element array beam pattern. The gain is calculated based on the element
    
    Args:
        num_col_el (int): Number of elements in the column.
        num_row_el (int): Number of elements in the row.
        element_spacing (float): Spacing between the elements.
        center_frequency (int): Center frequency in Hz.
        stepsize (float): Step size in degrees for the calculation.

    """
    def __init__(self, num_col_el: int, num_row_el: int, element_spacing: float, center_frequency: int):
        self.num_col_el = num_col_el
        self.num_row_el = num_row_el
        self.element_spacing = element_spacing
        self.center_frequency = center_frequency
    
    def calculateGain_Linear(self, theta: float, phi: float): #michael: theta_az,  theta_el? Difference=         
        # Constants
        c = 3e8  # Speed of light in m/s
        wavelength = c / self.center_frequency
        k = 2 * np.pi / wavelength  # Wavenumber
        I = 1
        beta_x = 0
        beta_y = 0        
        # Element positions and phase shifts
        d_x = self.element_spacing
        d_y = self.element_spacing
        #theta = self._theta_conversion(theta_az, theta_el) #michael: ?
        #phi = self._phi_conversion(theta_az, theta_el)     #michael: ?     
        af_sum = 0
        for n in range(1, self.num_col_el+1):
            sum_m = 0
            for m in range(1, self.num_row_el+1):
                sum_m += I * np.exp(1j * (m-1) * (k*d_x * np.sin(theta) * np.cos(phi) + beta_x))
            af_sum += I * sum_m * np.exp(1j * (n-1) * (k*d_y * np.sin(theta) * np.sin(phi) + beta_y))
            
        af_magnitude = np.abs(af_sum)
        #af_dB = 10 * np.log10(af_magnitude)
    
        return af_magnitude # michael: return linear here

    
    def _theta_conversion(self, theta_az, theta_el):
        return np.arccos(np.cos(theta_az) * np.sin(theta_el))

    def _phi_conversion(self, theta_az, theta_el):
        return np.arctan2(np.sin(theta_el), np.sin(theta_az) * np.cos(theta_el))  


class RFSource():
    r"""Implementation of a RF IQ sample source. This object polls IQ samples in a separate thread
    from a generator function (inputfct) that is provided by the user. Implementation is optimized
    for handling all kinds of shapes and sizes of the user-defined generator function's output by 
    using a custom written circular buffer 'BlockingCircularBuffer' defined in Utils.py.

    Args:
        inputfct (Callable): Generator function that generates IQ samples. Must output a list of np.array of 
            complex datatype.

    """
    def __init__(self, inputfct):
        self._input = inputfct

    def send(self, data: np.ndarray):
        r"""Pushes IQ samples to the internal circular buffer.
        
        Args:
            data (np.ndarray): IQ samples to push to the buffer.
        """
        self._buffer.bulk_put(data=data)
    
    def _initBuffer(self, buffersize):
        self._buffer = BlockingCircularBuffer(maxsize=buffersize, dtype=complex)
    
    def _get(self, numsamples: int):
        iq = self._buffer.bulk_get(n=numsamples)
        #print("RFSource _get: Pulled", len(iq), "samples.")
        return iq
    
    def _start(self,):
        assert self._input != None, "RFSource's input function is not defined. Use RFSource.setInput()"
        def work():
            while True:
                data = self._input()
                self.send(data=data)
        threading.Thread(target=work).start()
                    

class RFSink():
    r"""Implementation of a RF IQ sample sink. This object is called when processed IQ samples are available
    from the system and can be further processed by the user. A user-defined callable (outputfct) is called.
    Similar to the RFSource, the RFSink uses a custom written circular buffer 'BlockingCircularBuffer' 
    which is filled in a sepearte thread.
    
    Args:
        outputfct (Callable): Consumer function that coinsumes IQ samples. Must accept a np.array of 
            complex datatype.

    """
    def __init__(self, outputfct):
        self._output = outputfct

    def receive(self, numsamples: int) -> np.ndarray:
        r"""Receives IQ samples from the internal circular buffer.
        
        Args:
            numsamples (int): Number of samples to receive.

        Returns:
            np.ndarray: Numpy Array of processed IQ samples.
        """
        # receive numsamples amount of samples
        iq = self._buffer.bulk_get(n=numsamples)
        return iq

    def _initBuffer(self, buffersize, outputbatchsize):
        self._output_batchsize = outputbatchsize
        self._buffer = BlockingCircularBuffer(maxsize=buffersize, dtype=complex)
    
    def _push(self, iq: np.ndarray):
        self._buffer.bulk_put(data=iq)        
        #print("RFSink _push: Received ", iq)
    
    def _start(self,):
        assert self._output != None, "RFSink's output function is not defined. Use RFSource.setOutput()"
        def work():
            while True:
                data = self.receive(numsamples=self._output_batchsize)
                self._output(data)
        threading.Thread(target=work).start()

class RFEmulator(RequireEncode):
    r"""Encodable object for handling RF IQ link emulation.

    Functionality:
    - Emulation of RF links between ground stations and satellites (Doppler Effect, Path Loss, Athmospheric Effects, etc.).
    - Emulation of all cross RF links between ground station groups and satellite groups.
    - Control of visualization of RF links in the frontend.
    - Stream of IQ signal information (spectrogram, link properties [link budget,
        elevation angle, distance, etc.]) to frontend via websocket.
    
    Notes:

    Link Emulation is encapsuled into 'LinkEnvironment' objects that hold high-level information such as 
    ground and space object(s), and 'Link' objects that hold low-level information such as antenna and
    beam selection, IQ samples rate, transmit power, bandwidth, frequency, loss characteristics, min.
    elevation angles, etc. This is implemented mainly for optimization purposes, e.g., computing position
    and velocity relationships between communicating objects only once and re-using them in the low-level links.

    Todo: Complete documentation
    """

    def __init__(self,):
        self._linkenvs_earthtospace = []
        self._linkenvs_earthtospacegroup = []
    
    class LinkData(Enum):
        NONE = 0
        SPECTROGRAM = 1
        RAWIQ = 2


    def getLinkEnvironments(self,) -> List:
        r"""Return all registered link environments.
        
        Returns:
            List: List of all registered link environments, multiple data types.
        """
        return list(chain(self._linkenvs_earthtospace, self._linkenvs_earthtospacegroup))

    def emulateLinkGroundToSpace(self, groundObject: 'Stationary', spaceObject: 'Propagatable') -> 'LinkEnvironmentGroundToSpace':
        r"""Registers and returns a singular high-level link environment between ground and space object.
        
        Args:
            groundObject (Stationary): Ground object.
            spaceObject (Propagatable): Space object.

        Returns:
            LinkEnvironmentGroundToSpace: Registered link environment
        """
        # ensure objects have rf frontend specific
        assert groundObject.getRFFrontend() != None, "groundObject must have its RFFrontend specified."
        assert spaceObject.getRFFrontend() != None, "spaceObject must have its RFFrontend specified."
        #todo: if scenario has no frontend ignore visualize True in oninit..
        linkenvironment = LinkEnvironmentGroundToSpace(rfemulator=self, groundObject=groundObject, spaceObject=spaceObject)
        self._linkenvs_earthtospace.append(linkenvironment)
        return linkenvironment

    def emulateLinkGroundGroupToSpaceGroup(self, groundObjects: Union['Stationary', List['Stationary']], spaceGroup: 'PropagatableGroup') -> 'LinkEnvironmentGroundGroupToSpaceGroup':
        r"""Registers and returns a high-level link environment group between ground and space object groups.
        
        Args:
            groundObjects (Union[Stationary, List[Stationary]]): Ground object or list of ground objects of type Stationary
            spaceGroup (PropagatableGroup): Space object group of type PropagatableGroup

        Return:
            LinkEnvironmentGroundGroupToSpaceGroup: Registered link environment group
        """
        # Assertion for component correctness done in LinkEnvironmentGroundToSpaceGroup
        linkenvironment = LinkEnvironmentGroundGroupToSpaceGroup(rfemulator=self,groundObjects=groundObjects, spaceGroup=spaceGroup)
        self._linkenvs_earthtospacegroup.append(linkenvironment)
        return linkenvironment

    def onInitialize(self, scenario: 'Scenario'):
        # Internal method to be called on initialization of the RFEmulator.
        self._visualization = False
        for linkEnv in self.getLinkEnvironments(): # initialize all links separately
            linkEnv._initialize(scenario)
            self._visualization = self._visualization or any(l._IQvisualize for l in linkEnv.getLinks())
        self._visualization = self._visualization and scenario.hasFrontend()
        if self._visualization:
            threading.Thread(target=self._startRFWebSocket, args=()).start()

    async def _processLinkEnvironments(self, scenario: 'Scenario'):
        for linkEnv in self.getLinkEnvironments():
            await linkEnv._timestep_generic(scenario)
        for linkEnv in self.getLinkEnvironments():
            await linkEnv._timestep_iq(scenario)
        # process IQ emulation optimized:


    def visualizationActive(self,) -> bool:
        r"""Returns the visualization state of the RFEmulator.
        
        Returns:
            bool: Visualization state.
        """
        return self._visualization

    def encode(self, messagetype: Message.Type) -> Dict:
        # Internal method for encoding the RFEmulator object for transmission to the frontend.
        if messagetype == Message.Type.SCENARIOSETUP:
            return {
            "parent_type": 'RFEmulator',
            "object_type": self.__class__.__name__,
            "linkEnvironments": self.getLinkEnvironments(),
            }
        elif messagetype == Message.Type.PROPAGATIONDATA:
            return {
            "parent_type": 'RFEmulator',
            "object_type": self.__class__.__name__,
            "linkEnvironments": self.getLinkEnvironments(),
            }
        else:
            raise ValueError("Not implemented yet.")


    # Websocket handling for managing RF data transfer between backend and frontend.
    def _startRFWebSocket(self, ):
        async def _server(websocket):
            self._rfwebsocket = websocket
            async for message in websocket:
                #logging.info("[Websocket] Message received: %s", message)
                if (message in self.clientMessageSemaphores):
                    self.clientMessageSemaphores[message].release()
        self.clientMessageSemaphores = {}
        self._temp = 0
        PORT_WEBSOCKET = 8766
        logging.debug("RF Websocket listening at localhost:%s", str(PORT_WEBSOCKET))
        self.loop = asyncio.new_event_loop()
        async def coro():
            await serve(_server, "localhost", PORT_WEBSOCKET)
        self.loop.run_until_complete(coro())
        self.loop.run_forever()
        logging.warning("RF Websocket finished. This should never occur.")
    
    async def _sendWSMessage(self, message: str):
        logging.debug("Sending message to ws client %s", message)
        assert self._rfwebsocket != None, "Cannot send message, websocket connection not established yet."
        asyncio.run_coroutine_threadsafe(self._rfwebsocket.send(message), self.loop) # todo check if await is needed

    async def _awaitWSMessage(self, message: str):
        semaphore = threading.Semaphore(1)
        semaphore.acquire()
        self.clientMessageSemaphores[message] = semaphore
        semaphore.acquire()
        self.clientMessageSemaphores.pop(message)
    async def _sendLinkData(self, link, data):
        # Encode and pack link data to binary format and send it to the frontend via websocket.
        if not self._visualization:
            return
        self._temp = self._temp + 2
        link_id_bytes = link._id.encode('utf-8')
        link_viz_type_bytes = struct.pack('<I', link._IQvisualize_type.value) #<I = uint32
        data_bytes = data.tobytes()
        message = link_id_bytes + link_viz_type_bytes + data_bytes
        await self._sendWSMessage(message)


class LinkDirection(Enum):
    DOWNLINK = 1,
    UPLINK = 2


class LinkLevelGroundToSpace():
    r"""Linklevel object definition that provides low-level link-related functionality.

    Functionality:
    - Compute atmospheric losses based on 3GPP TR 38.811 Release 15.
    - Compute shadowing losses
    - compute scintillation losses
    - utility functions for computing elevation angles, distance, etc.
    - compute geostationary avoidance angle and regions
    - compute EIRP
    - compute G/T
    - compute complete link budget

    Args:
        link (LinkBase): Link object that holds high-level link information.

    Note:

    This object is used internally.

    Todo: Complete function documentation.
    """

    def __init__(self, link: 'LinkBase'):
        self._link = link

    def getLink(self,) -> 'LinkBase':
        r"""Returns the parent link object."""
        return self._link

    def getAtmosphericLosses(self) -> float:
        r"""Compute atmospheric losses in dB based on 3GPP TR 38.811 Release 15.
        
        Returns:
            float: Atmospheric losses in dB.
        """
        # based on 3GPP TR 38.811 Release 15
        if self.getLink().getLinkDirection() == LinkDirection.DOWNLINK:
            #Downlink
            atmosphericLoss = 0
            if self.getLink().getCenterFrequency() > 1e09: # todo michael: should link have one center frequency or really both ground/space beam?           
                zenithAtten = np.array([0.0350, 0.0215, 0.0114, 0.0049, 0.0021, 0.0034, 0.0088, 0.0186, 0.0329, 0.0520, 0.0830,
                    0.1280, 0.1828, 0.2434, 0.3058, 0.3657, 0.4192, 0.4622, 0.4906, 0.5003, 0.4937, 0.4750,
                    0.4473, 0.4135, 0.3766, 0.3395, 0.3053, 0.2768, 0.2571, 0.2488, 0.2493, 0.2564, 0.2688,
                    0.2854, 0.3049, 0.3263, 0.3482, 0.3697, 0.3894, 0.4068, 0.4269, 0.4526, 0.4857, 0.5279,
                    0.5809, 0.6464, 0.7261, 0.8219, 0.9354, 1.3915, 2.4932, 3.8721, 7.2999, 13.9536, 32.6008,
                    69.8602, 119.6762, 175.2541, 229.7990, 282.8459, 281.2638, 144.6855, 83.7671, 60.7415, 41.3750, 25.7806,
                    14.0715, 6.3608, 2.7617, 2.3405, 2.0947, 1.8376, 1.5785, 1.3268, 1.0919, 0.8831, 0.7099,
                    0.5815, 0.5075, 0.4740, 0.4381, 0.3998, 0.3609, 0.3228, 0.2871, 0.2555, 0.2294, 0.2106,
                    0.2005, 0.1961, 0.1915, 0.1872, 0.1835, 0.1809, 0.1799, 0.1808, 0.1842, 0.1904, 0.2000
                ])
                idxFreq = int(np.round(self.getLink().getCenterFrequency()/1e09))
                tempAtten = zenithAtten[idxFreq]
                elevAngle = self.getElevationAngle()
                # check if elevation angle is below minimum elevation defined. This avoids a) reaching infinite b) get negative losses c) further calcs as link is anyway not interesting
                if elevAngle < self.getLink().getMinElev():
                    atmosphericLoss = tempAtten/np.sin(self.getLink().getMinElev()/180*np.pi)
                else:
                    atmosphericLoss = tempAtten/np.sin(elevAngle/180*np.pi)
            return atmosphericLoss
        else:
            #Uplink
            atmosphericLoss = 0
            if self.getLink().getCenterFrequency() > 1000000000:            
                zenithAtten = np.array([0.0350, 0.0215, 0.0114, 0.0049, 0.0021, 0.0034, 0.0088, 0.0186, 0.0329, 0.0520, 0.0830,
                    0.1280, 0.1828, 0.2434, 0.3058, 0.3657, 0.4192, 0.4622, 0.4906, 0.5003, 0.4937, 0.4750,
                    0.4473, 0.4135, 0.3766, 0.3395, 0.3053, 0.2768, 0.2571, 0.2488, 0.2493, 0.2564, 0.2688,
                    0.2854, 0.3049, 0.3263, 0.3482, 0.3697, 0.3894, 0.4068, 0.4269, 0.4526, 0.4857, 0.5279,
                    0.5809, 0.6464, 0.7261, 0.8219, 0.9354, 1.3915, 2.4932, 3.8721, 7.2999, 13.9536, 32.6008,
                    69.8602, 119.6762, 175.2541, 229.7990, 282.8459, 281.2638, 144.6855, 83.7671, 60.7415, 41.3750, 25.7806,
                    14.0715, 6.3608, 2.7617, 2.3405, 2.0947, 1.8376, 1.5785, 1.3268, 1.0919, 0.8831, 0.7099,
                    0.5815, 0.5075, 0.4740, 0.4381, 0.3998, 0.3609, 0.3228, 0.2871, 0.2555, 0.2294, 0.2106,
                    0.2005, 0.1961, 0.1915, 0.1872, 0.1835, 0.1809, 0.1799, 0.1808, 0.1842, 0.1904, 0.2000
                ])
                idxFreq = int(np.round(self.getLink().getCenterFrequency()/1e09))
                tempAtten = zenithAtten[idxFreq]
                elevAngle = self.getElevationAngle()
                if elevAngle < self.getLink().getMinElev():
                    atmosphericLoss = tempAtten/np.sin(self.getLink().getMinElev()/180*np.pi)
                else:
                    atmosphericLoss = tempAtten/np.sin(elevAngle/180*np.pi)
            return atmosphericLoss
        
    def getShadowLoss(self) -> float:
        r"""Compute shadowing losses in dB based on 3GPP TR 38.811 Release 15.
        
        Returns:
            float: Shadowing losses in dB.
        """
        # based on Table 6.6.2-1: Shadow Fading and Clutter Losses for different environments
        env_temp = self.getLink().getShadowScenario()
        if isinstance(env_temp,str):
            if self.getLink().getCenterFrequency() < 6e09:
                if env_temp == "Dense Urban":
                    SF_durban_var = [3.5,3.4,2.9,3.0,3.1,2.7,2.5,2.3,1.2]
                    el_temp = int(np.round(self.getElevationAngle()/10))
                    LossSF = abs(np.random.normal(0,SF_durban_var[el_temp]))
                elif env_temp == "Urban":
                    SF_urban_var = [4,4,4,4,4,4,4,4,4]
                    el_temp = int(np.round(self.getElevationAngle()/10))
                    LossSF = abs(np.random.normal(0,SF_urban_var[el_temp]))
                elif (env_temp == "Suburban" or env_temp == "Rural"):
                    SF_subrural_var = [1.79,1.14,1.14,0.92,1.42,1.56,0.85,0.72,0.72]
                    el_temp = int(np.round(self.getElevationAngle()/10))
                    LossSF = abs(np.random.normal(0,SF_subrural_var[el_temp]))
                else:
                    LossSF = 0
            else:
                if env_temp == "Dense Urban":
                    SF_durban_var = [2.9,2.4,2.7,2.4,2.4,2.7,2.6,2.8,0.6]
                    el_temp = int(np.round(self.getElevationAngle()/10))
                    LossSF = abs(np.random.normal(0,SF_durban_var[el_temp]))
                elif env_temp == "Urban":
                    SF_urban_var = [4,4,4,4,4,4,4,4,4]
                    el_temp = int(np.round(self.getElevationAngle()/10))
                    LossSF = abs(np.random.normal(0,SF_urban_var[el_temp]))
                elif (env_temp == "Suburban" or env_temp == "Rural"):
                    SF_subrural_var = [1.9,1.6,1.9,2.3,2.7,3.1,3.0,3.6,0.4]
                    el_temp = int(np.round(self.getElevationAngle()/10))
                    LossSF = abs(np.random.normal(0,SF_subrural_var[el_temp]))
                else:
                    LossSF = 0
        else:
            LossSF = 0

        return LossSF

    def getScintallationLoss(self) -> float:
        r"""Compute scintillation losses in dB based on 3GPP TR 38.811 Release 15.
        
        Returns:
            float: Scintillation losses in dB.
        """
        # Scintallation Losses based on groundstation/UE latitudes
        lat_temp = self.getLink().getGroundObject()._stationaryposition.lat
        baseline_loss = 1.1 #dB and 99% that ordinate is not exceeded in Fig. 6.6.6.1.4
        freq_temp = self.getLink().getCenterFrequency()
        if abs(lat_temp) < 20:
            LossScint = 1.1*(freq_temp/4)**-1.5/np.sqrt(2)
        elif abs(lat_temp) >= 20 or abs(lat_temp) < 60:
            LossScint = 0
        else:
            LossScint = 0
        return LossScint

    def getElevationAngle(self,) -> float:
        r"""Compute elevation angle in degrees.
        
        Returns:
            float: Elevation angle in degrees.
        """
        satPos = self.getLink().getSpaceObject().getPosition()
        groundPos = self.getLink().getGroundObject().getPosition()
        dirVec = np.subtract(groundPos,satPos)
        dirVecGround = dirVec/np.linalg.norm(dirVec)
        vGround = groundPos/np.linalg.norm(groundPos)
        elevAngle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(dirVecGround,vGround)),np.dot(dirVecGround,vGround)))-90
        return elevAngle

    def getAngleSat(self,) -> float:
        r"""Utility function: Compute angle between satellite and ground station in degrees.
        
        Returns:
            float: Angle in degrees.
        """
        satPos = self.getLink().getSpaceObject().getPosition()
        groundPos = self.getLink().getGroundObject().getPosition()
        dirVec = np.subtract(satPos,groundPos) # i switched the arguments
        nom = np.dot(groundPos, dirVec)
        den =  np.sqrt(np.sum(groundPos**2)*np.sum(dirVec**2))
        el = np.arccos(nom/den)*180/np.pi
        el_ang = 90 - el

        return el_ang

    def checkCoverage(self,) -> bool:
        r"""Utility function: Check if satellite is in coverage of the ground station.
        
        Returns:
            bool: True if satellite is in coverage, False otherwise.
        """
        commDistance = pdist2(self.getLink().getSpaceObject().getPosition(),self.getLink().getGroundObject().getPosition())
        slantRange = getSlantRange(self.getLink().getSpaceObject().getAltitude(),self.getLink().getMinElev())

        return commDistance < slantRange
    
    def checkGSO(self,GSO_avoidance_angle) -> bool:
        r"""Utility function for checking if satellite is in GSO avoidance angle.
        
        Args:
            GSO_avoidance_angle (float): GSO avoidance angle in degrees.

        Returns:
            bool: True if satellite is in GSO avoidance angle, False otherwise
        """
        # assuming a GEO directly above the link groundstation
        # ToDo: Maybe add a list of all GEO satellites, right now monitors worst case
        RE = 6374 #km
        h_GEO = 35786 #km
        r_GEO = RE + h_GEO
        #define a GSO above groundstation position
        gs_pos = self.getLink().getGroundObject().getPosition()
        gs_lon =  np.arctan2(gs_pos[1],gs_pos[0])
        sat_pos = self.getLink().getSpaceObject().getPosition()
        GEO_pos = [r_GEO*np.cos(gs_lon/180*np.pi), r_GEO*np.sin(gs_lon), 0]
        dir_vec = np.subtract(gs_pos,sat_pos)
        dir_vec_norm = dir_vec/np.linalg.norm(dir_vec)
        dir_vec_GEO = np.subtract(gs_pos,GEO_pos)
        dir_vec_GEO_norm = dir_vec_GEO/np.linalg.norm(dir_vec_GEO)
        real_angle = np.arccos(np.dot(dir_vec_norm,dir_vec_GEO_norm)/(np.linalg.norm(dir_vec_norm)*np.linalg.norm(dir_vec_GEO_norm)))
        GSO_flag = real_angle > (GSO_avoidance_angle/180*np.pi)
        return GSO_flag


    def getOffsideAngle(self) -> float:
        r"""Utility function: Compute offside angle in degrees.
        
        Returns:
            float: Offside angle in degrees.
        """
        if self.getLink().getLinkDirection() == LinkDirection.DOWNLINK:
            sat_track_obj = self.getLink().getGroundBeam().orientation()._lookAtTrackObj
            sat_link_obj = self.getLink().getSpaceObject()
            if sat_track_obj == None or sat_track_obj == sat_link_obj:
                delta_theta = 0
                delta_phi = 0
            else:
                    track_obj_pos = sat_track_obj.getPosition()
                    comm_obj_posA = sat_link_obj.getPosition()
                    comm_obj_posB = self.getLink().getGroundObject().getPosition()
                    ###
                    # Calculate vector from communication link 
                    r_B = np.subtract(comm_obj_posA,comm_obj_posB)
                    # Calculate vector from tracked link
                    r_A = np.subtract(track_obj_pos,comm_obj_posB)

                    # Calculate azimuth (theta) and elevation (phi) for both vectors
                    theta_B = np.degrees(np.arctan2(r_B[1], r_B[0]))
                    phi_B = np.degrees(np.arctan2(r_B[2], np.sqrt(r_B[0]**2 + r_B[1]**2)))
                    theta_A = np.degrees(np.arctan2(r_A[1], r_A[0]))
                    phi_A = np.degrees(np.arctan2(r_A[2], np.sqrt(r_A[0]**2 + r_A[1]**2)))

                    # Calculate misalignment
                    delta_theta = np.abs(theta_B - theta_A)
                    delta_phi = np.abs(phi_B - phi_A)
        elif self.getLink().getLinkDirection() == LinkDirection.UPLINK:  
            ground_track_obj = self.getLink().getSpaceBeam().orientation()._lookAtTrackObj
            ground_link_obj = self.getLink().getGroundObject()
            if ground_track_obj == None or ground_track_obj == ground_link_obj:
                delta_theta = 0
                delta_phi = 0
            else:
                    track_obj_pos = ground_track_obj.getPosition()
                    comm_obj_posA = ground_link_obj.getPosition()
                    comm_obj_posB = self.getLink().getSpaceObject().getPosition()
                    ###
                    # Calculate vector from communication link 
                    r_B = np.subtract(comm_obj_posA,comm_obj_posB)
                    # Calculate vector from tracked link
                    r_A = np.subtract(track_obj_pos,comm_obj_posB)

                    # Calculate azimuth (theta) and elevation (phi) for both vectors
                    theta_B = np.degrees(np.arctan2(r_B[1], r_B[0]))
                    phi_B = np.degrees(np.arctan2(r_B[2], np.sqrt(r_B[0]**2 + r_B[1]**2)))
                    theta_A = np.degrees(np.arctan2(r_A[1], r_A[0]))
                    phi_A = np.degrees(np.arctan2(r_A[2], np.sqrt(r_A[0]**2 + r_A[1]**2)))

                    # Calculate misalignment
                    delta_theta = np.abs(theta_B - theta_A)
                    delta_phi = np.abs(phi_B - phi_A)
        #print(delta_phi)
        #print(delta_theta)
        return delta_theta,delta_phi

    def getEIRP(self) -> float:
        r"""Compute EIRP in dB.
        
        Returns:
            float: EIRP in dB.
        """ 
        if self.getLink().getLinkDirection() == LinkDirection.DOWNLINK:
            #[theta,phi] = self.getOffsideAngle()
            # here consider pointing always on target
            theta = 0
            phi = 0
            ant_gain = self.getLink().getSpaceBeam()._pattern.calculateGain_Linear(theta=theta,phi=phi) 
            tx_pow = self.getLink().getSpaceBeam().getTransmitPower()
            eirp = ant_gain + 10*np.log10(tx_pow)
        else:
            # here consider pointing always on target
            theta = 0
            phi = 0
            ant_gain = self.getLink().getGroundBeam()._pattern.calculateGain_Linear(theta=theta,phi=phi)
            tx_pow = self.getLink().getGroundBeam().getTransmitPower()
            eirp = ant_gain+10*np.log10(tx_pow)

        return eirp

    def getGT(self) -> float:
        r"""Compute G/T in dB.
        
        Returns:
            float: G/T in dB.
        """
        if self.getLink().getLinkDirection() == LinkDirection.DOWNLINK:
            [theta,phi] = self.getOffsideAngle()
            ant_gain = self.getLink().getGroundBeam()._pattern.calculateGain_Linear(theta=theta,phi=phi)
            NF = self.getLink().getGroundBeam().getAntenna().getNoiseFigure()
            T_amb = self.getLink().getGroundBeam().getAntenna().getAmbientTemperature()
            T_ant = self.getLink().getGroundBeam().getAntenna().getAntennaTemperature()
            if NF is None or T_amb is None or T_ant is None:
                G_T = ant_gain - 0
            else:
                G_T = ant_gain - NF - 10*np.log10(T_amb+(T_ant-T_amb)*10**(-0.1*NF))
        else:
            [theta,phi] = self.getOffsideAngle()
            ant_gain = self.getLink().getSpaceBeam()._pattern.calculateGain_Linear(theta=theta,phi=phi)
            NF = self.getLink().getSpaceBeam().getAntenna().getNoiseFigure()
            T_amb = self.getLink().getSpaceBeam().getAntenna().getAmbientTemperature()
            T_ant = self.getLink().getSpaceBeam().getAntenna().getAntennaTemperature()
            if NF is None or T_amb is None or T_ant is None:
                G_T = ant_gain - 0
            else:
                G_T = ant_gain - NF - 10*np.log10(T_amb+(T_ant-T_amb)*10**(-0.1*NF))
        return G_T

    def calculateLinkBudget(self, returnAll: Optional[bool] = False) -> float:
        r"""Compute the complete link budget in dB.
        
        Args:
            returnAll (Optional[bool]): If True, return additional data, i.e., distance, elevation, 
            antenna gains of space- and ground-based station.

        Returns:
            float: Complete link budget in dB.
        """
        if self.getLink().getLinkDirection() == LinkDirection.DOWNLINK:
            K = -228.6
            RE = 6374
            #Propagation Calcs
            distance = pdist2(self.getLink().getGroundObject().getPosition(), self.getLink().getSpaceObject().getPosition())
            losses = self.getAtmosphericLosses() + self.getShadowLoss() + self.getScintallationLoss()
            #Transmit Side
            eirp = self.getEIRP()
            path_loss = 22+20*np.log10(distance*1000/(3e08/self.getLink().getCenterFrequency()))
            #Receiver Side
            G_T = self.getGT()
            bandwidth_noise = 10*np.log10(self.getLink().getSpaceBeam().getTransmitBandwidth())
            #Received Signal to Noise Ratio
            snr = eirp + G_T - path_loss - losses - K - bandwidth_noise
        else:
            K = -228.6
            RE = 6374
            #Propagation Calcs
            distance = pdist2(self.getLink().getGroundObject().getPosition(),self.getLink().getSpaceObject().getPosition())
            losses = self.getAtmosphericLosses() + self.getShadowLoss() + self.getScintallationLoss()
            #Transmit Side
            eirp = self.getEIRP()
            path_loss = 22+20*np.log10(distance*1000/(3e08/self.getLink().getCenterFrequency()))
            #Receiver Side
            G_T = self.getGT()
            bandwidth_noise = 10*np.log10(self.getLink().getGroundBeam().getTransmitBandwidth())
            #Received Signal to Noise Ratio
            snr = eirp + G_T - path_loss - losses - K - bandwidth_noise

        if not returnAll:
            return snr
        else:
            data = {
                "snr": snr,
                "distance": distance,
                "elevation": self.getElevationAngle(),
                "gsat": 10*np.log10(self.getLink().getSpaceBeam().calculateGain(target=self.getLink().getGroundObject().getPosition())),
                "ggnd": 10*np.log10(self.getLink().getGroundBeam().calculateGain(target=self.getLink().getSpaceObject().getPosition())),
            }
            return data
        

class LinkBase(ABC):
    r"""Abstract base class defining standard interfaces for all low-level 'Link' classes.
    """

    @abstractmethod
    def getEnvironment(self,):
        pass
    
    def getLinkDirection(self,) -> LinkDirection:
        pass
    
    def getGroundObject(self,) -> 'Stationary':
        pass
    
    def getSpaceObject(self,) -> 'Propagatable':
        pass
    
    def getGroundBeam(self,) -> RFFrontend.Antenna.Beam:
        pass
    
    def getSpaceBeam(self,) -> RFFrontend.Antenna.Beam:
        pass
    
    def getCenterFrequency(self,) -> float:
        pass

    def getShadowScenario(self,) -> Optional[str]:
        pass

    def getMinElev(self,) -> float:
        pass


class LinkEnvironmentGroundToSpace(RequireEncode):
    r"""Implementation of the high-level link environment object for a singular ground-to-space link.
    This class is internally used by the RFEmulator object. The instantiated object holds information
    that is shared between all low-level links that are subcomponents of this link environment.

    Args:
    - rfemulator: RFEmulator object
    - groundObject: Stationary object
    - spaceObject: Propagatable object

    """

    def __init__(self, rfemulator: RFEmulator, groundObject, spaceObject):
        self._rfemulator = rfemulator
        self._id = randUUID()
        self._groundObject = groundObject
        self._spaceObject = spaceObject
        self._links = []
        # operating parameters
        self._initialized = False
        self._time_start = None # start of timeframe
        self._groundObject_start = None # GroundObject pos+vel at previous timestep
        self._groundObject_end = None # GroundObject pos+vel at current timestep
        self._time_end = None   # end of timeframe
        self._spaceObject_start = None # SpaceObject pos+vel at previous timestep
        self._spaceObject_end = None   # SpaceObject pos+vel at current timestep
        # self timing

    class Link(RequireEncode, LinkBase):
        r"""Implementation of the low-level link object for a singular ground-to-space link.
        This class is internally used by the LinkEnvironmentGroundToSpace object. The instantiated object holds
        information that is specific to the link.
        
        Args:
        - environment: Specifies the Parent LinkEnvironmentGroundToSpace object
        - direction: Specifies the link direction (downlink/uplink)
        - groundBeam: Specifies the ground beam object
        - spaceBeam: Specifies the space beam object
        - centerFrequency: Specifies the center frequency of the link
        - shadowScenario: Specifies the shadow scenario
        - minElevation: Specifies the minimum elevation angle
        - label: Specifies the label of the link to be displayed in the frontend
        - visualizeLinkStats: Specifies if link stats should be visualized in the frontend

        """
        def __init__(self, environment: 'LinkEnvironmentGroundToSpace', direction: LinkDirection, groundBeam: RFFrontend.Antenna.Beam, spaceBeam: RFFrontend.Antenna.Beam, centerFrequency: float, shadowScenario: Optional[str]=None, minElevation: Optional[float]=5, label: Optional[str]=None, visualizeLinkStats: Optional[bool]=True):
            self._id = randUUID()
            self._environment = environment
            self._direction = direction
            self._emulateIQ = False
            self._IQvisualize = False
            self._IQvisualize_type = None
            self._groundBeam = groundBeam
            self._spaceBeam = spaceBeam
            self._centerFrequency = centerFrequency
            self._shadowScenario = shadowScenario
            self._minElevation = minElevation 
            self._label = label
            self._visualizeLinkStats = visualizeLinkStats
            self._linklevel = LinkLevelGroundToSpace(link = self)

        def getEnvironment(self,):
            r"""Return the parent LinkEnvironmentGroundToSpace object.
            
            Returns:
                LinkEnvironmentGroundToSpace: Parent object.
            """
            return self._environment
        
        def getLinkDirection(self,) -> LinkDirection:
            r"""Return the link direction.
            
            Returns:
                LinkDirection: Link direction.
            """
            return self._direction
        
        def getGroundObject(self,) -> 'Stationary':
            r"""Return the ground object.
            
            Returns:
                Stationary: Ground object.
            """
            return self.getEnvironment().getGroundObject()
        
        def getSpaceObject(self,) -> 'Propagatable':
            r"""Return the space object.
            
            Returns:
                Propagatable: Space object.
            """
            return self.getEnvironment().getSpaceObject()
        
        def getGroundBeam(self,) -> RFFrontend.Antenna.Beam:
            r"""Return the ground beam object.
            
            Returns:
                RFFrontend.Antenna.Beam: Ground beam object.
            """
            return self._groundBeam
        
        def getSpaceBeam(self,) -> RFFrontend.Antenna.Beam:
            r"""Return the space beam object.
            
            Returns:
                RFFrontend.Antenna.Beam: Space beam object
            """
            return self._spaceBeam
        
        def getCenterFrequency(self,) -> float:
            r"""Return the center frequency.
            
            Returns:
                float: Center frequency.
            """
            return self._centerFrequency

        def getShadowScenario(self,) -> Optional[str]:
            r"""Return the shadow scenario.
            
            Returns:
                Optional[str]: Shadow scenario type as string.
            """
            return self._shadowScenario

        def getMinElev(self,) -> float:
            r"""Return the minimum elevation angle.
            
            Returns:
                float: Minimum elevation angle.
            """
            return self._minElevation

        def _getSource(self,) -> RFSource:
            r"""Return the RFSource object.
            
            Return:
                'RFSource': RFSource object associated with the link's IQ emulation.
            """
            assert self._emulateIQ == True, 'Function only available if IQ emulation activated'
            return self._IQRfsource

        def _getSink(self, ) -> RFSink:
            r"""Return the RFSink object.
            
            Returns:
                'RFSink': RFSink object associated with the link's IQ emulation.
            """
            assert self._emulateIQ == True, 'Function only available if IQ emulation activated'
            return self._IQRfsink
        
        def getSampleRate(self,) -> int:
            r"""Return the sample rate of the link.
            
            Returns:
                int: Sample rate.
            """
            assert self._emulateIQ == True, 'Function only available if IQ emulation activated'
            return self._IQSampleRate
        
        def getTiming(self,) -> RealTiming:
            r"""Return the timing object of the link. Used for internal timing and performance optimization purposes.
            
            Returns:
                RealTiming: Timing object.
            """
            assert self._emulateIQ == True, 'Function only available if IQ emulation activated'
            return self._IQTiming
        
        def emulateIQ(self, sampleRate, rfsourcefct, rfsinkfct, sinkBatchSize: Optional[int] = None, bufferSize: Optional[int]=5, visualize: Optional[bool] = False, onProcessed : Optional[bool] = None):
            r"""Activate IQ emulation for the link with the specified parameters
            
            Args:
            - sampleRate: Sample rate in Hz
            - rfsourcefct: Function that generates IQ samples
            - rfsinkfct: Function that processes IQ samples
            - sinkBatchSize: Batch size of the sink
            - bufferSize: Multiplier for calculating the buffer size of the internal IQ buffers
            - visualize: If True, visualize the IQ data in the frontend
            - onProcessed: Callable that will be invoked when a IQ emulation step is concluded

            """
            self._emulateIQ = True
            self._IQSampleRate = sampleRate
            self._IQRfsource = RFSource(inputfct=rfsourcefct)
            self._IQRfsink = RFSink(outputfct=rfsinkfct)
            self._IQsinkBatchSize = sinkBatchSize
            self._IQBufferSize = bufferSize
            self._IQvisualize = visualize
            self._IQvisualize_type = RFEmulator.LinkData.SPECTROGRAM if visualize else None
            self._IQOnProcessed = onProcessed
            self._IQTiming = RealTiming()
            if self._IQvisualize and RFEMULATOR_GPU_IQ_OFFLOAD_TENSORFLOW:
                getPyModule("tensorflow") # make sure it is imported
        
        def linklevel(self,) -> LinkLevelGroundToSpace:
            r"""Return the used link level object.
            
            Returns:
                LinkLevelGroundToSpace: Link level object used for defining link characteristics.
            """
            return self._linklevel
            
        def _initialize(self, scenario):
            timestep_secs = scenario.getTimestep().total_seconds()
            if self._emulateIQ:
                self._IQsinkBatchSize = self._IQsinkBatchSize if self._IQsinkBatchSize != None else int(self._IQSampleRate*timestep_secs) # one output per timestep
                self._IQRfsource._initBuffer(buffersize=int(self._IQSampleRate*timestep_secs*self._IQBufferSize))
                self._IQRfsink._initBuffer(buffersize=int(self._IQSampleRate*timestep_secs*self._IQBufferSize), outputbatchsize=self._IQsinkBatchSize)
                self._IQRfsource._start()
                self._IQRfsink._start()

        def _IQTimestep_prepare(self, scenario):
            if self._emulateIQ or self._visualizeLinkStats:
                # compute link level stats
                self._lastlinkleveldata = self.linklevel().calculateLinkBudget(returnAll=True)
                if self._emulateIQ:                    
                    self._lastlinkleveldata["processingrateiq"] = self._IQTiming.processedThroughput()

        
        async def _IQTimestep_iq(self, scenario, stepsize_sec: float):
            if self._emulateIQ:
                numsamps = np.array(np.round(self._IQSampleRate*stepsize_sec), dtype=np.int64) # todo fix for fractional # calculate number of samples to be processed in current timeframe
                iq_source = self._IQRfsource._get(numsamples=numsamps)
                iq_processed = self._processIQ(iq=iq_source, stepsize_sec=stepsize_sec)
                self._IQRfsink._push(iq=iq_processed)
                self._IQTiming.count(numsamps)
                if self._IQOnProcessed != None:
                    self._IQOnProcessed(self)
                # IQ data has been processed. Prepare for visualization as specific in self._IQvisualize_type and send to website:
                if self._environment._rfemulator._visualization and self._IQvisualize_type == RFEmulator.LinkData.SPECTROGRAM:
                    # calculate spectrum with fft (no windowing)
                    tf = getPyModule("tensorflow") if RFEMULATOR_GPU_IQ_OFFLOAD_TENSORFLOW else None                   
                    fft_max = 256
                    fft_len = fft_max if numsamps > fft_max else numsamps
                    fft = tf.signal.fft(iq_processed[:fft_len]) if RFEMULATOR_GPU_IQ_OFFLOAD_TENSORFLOW else np.fft.fft(iq_processed[:fft_len])
                    fft = fft if RFEMULATOR_GPU_IQ_OFFLOAD_TENSORFLOW else np.array(fft, dtype=np.complex64)
                    fft = tf.signal.fftshift(fft) if RFEMULATOR_GPU_IQ_OFFLOAD_TENSORFLOW else np.fft.fftshift(fft)
                    magnitude = tf.abs(fft) if RFEMULATOR_GPU_IQ_OFFLOAD_TENSORFLOW else np.abs(fft)
                    power = magnitude**2
                    power_mW = power * 1000
                    power_dBm = (10 * tf.math.log(power_mW, 10) / tf.math.log(10.0)).numpy() if RFEMULATOR_GPU_IQ_OFFLOAD_TENSORFLOW else 10 * np.log10(power_mW) #dtype=float32
                    await self._environment._rfemulator._sendLinkData(link=self, data=power_dBm)
                else:
                    pass
        
        def _processIQ(self, iq, stepsize_sec: float) -> np.ndarray:
            groundInfo = [self.getEnvironment()._groundObject_start, self.getEnvironment()._groundObject_end]
            spaceInfo = [self.getEnvironment()._spaceObject_start, self.getEnvironment()._spaceObject_end]
            #processedIQ_1 = applyDopplerIQEarthSpace(link=self, groundInfo=groundInfo, spaceInfo=spaceInfo, iq=iq, print1=True)
            processedIQ_2 = applyDopplerIQEarthSpaceEfficient(link=self, groundInfo=groundInfo, spaceInfo=spaceInfo, iq=iq)
            return processedIQ_2
        
        
        def encode(self, messagetype: Message.Type) -> Dict:
            # Internal method to encode the object for message passing
            if messagetype == Message.Type.SCENARIOSETUP:
                return {
                "parent_type": 'LinkEnvironmentGroundToSpace.Link',
                "object_type": self.__class__.__name__,
                "id": self._id,
                "label": self._label,
                "groundAntennaIdx": self._groundBeam.getAntenna()._antIdx,
                "groundBeamIdx": self._groundBeam._beamIdx,
                "spaceAntennaIdx": self._spaceBeam.getAntenna()._antIdx,
                "spaceBeamIdx": self._spaceBeam._beamIdx,
                "linkdirection": self._direction.name,
                "centerFrequency": self._centerFrequency,                
                "minElevation": self._minElevation,
                "iqstream": self._emulateIQ,
                **({"iq_samplerate": self._IQSampleRate} if self._emulateIQ else {}),
                **({"iq_visualize": self._IQvisualize} if self._emulateIQ else {}),
                "visualizeLinkStats": self._visualizeLinkStats,
                }
            elif messagetype == Message.Type.PROPAGATIONDATA:
                return {
                "id": self._id,
                **({"lastlinkleveldata": self._lastlinkleveldata} if hasattr(self, "_lastlinkleveldata") else {}),
                }
            else:
                raise ValueError("Not implemented yet.")
    
    def getLinks(self,) -> List['Link']:
        r"""Returns all registered links.
        
        Returns:
            List[Link]: List of all registered links.
        """
        return self._links

    def getGroundObject(self,) -> 'Stationary':
        r"""Returns the ground object.
        
        Returns:
            Stationary: Ground object.
        """
        return self._groundObject
    
    def getSpaceObject(self,) -> 'Propagatable':
        r"""Returns the space object.
        
        Returns:
            Propagatable: Space object.
        """
        return self._spaceObject

    def _initialize(self, scenario):
        for link in self._links:
            link._initialize(scenario=scenario)

    def registerLink(self, direction: LinkDirection, groundBeam: RFFrontend.Antenna.Beam, spaceBeam, centerFrequency: float, shadowScenario: Optional[str]=None, minElevation: Optional[float]=5, label: Optional[str]=None, visualizeLinkStats: Optional[bool]=True) -> Link:
        r"""Registers a new link to the environment.
        
        Args:
        - direction: Specifies the link direction (downlink/uplink)
        - groundBeam: Specifies the ground beam object
        - spaceBeam: Specifies the space beam object
        - centerFrequency: Specifies the center frequency of the link
        - shadowScenario: Specifies the shadow scenario
        - minElevation: Specifies the minimum elevation angle
        - label: Specifies the label of the link to be displayed in the frontend
        - visualizeLinkStats: Specifies if link stats should be visualized in the frontend

        Returns:
            Link: Registered link object.
        """

        link = LinkEnvironmentGroundToSpace.Link(environment = self, direction=direction, groundBeam=groundBeam, spaceBeam=spaceBeam, centerFrequency=centerFrequency, 
                                                shadowScenario=shadowScenario, minElevation=minElevation, label=label, visualizeLinkStats=visualizeLinkStats)
        self._links.append(link)
        return link  

    def encode(self, messagetype: Message.Type) -> Dict:    
        # Internal method for encoding the Linkenvironment object for message passing    
        if messagetype == Message.Type.SCENARIOSETUP:
            return {
            "parent_type": 'LinkEnvironmentGroundToSpace',
            "object_type": self.__class__.__name__,
            "id": self._id,
            "groundObject": self._groundObject.getId(),
            "spaceObject": self._spaceObject.getId(),
            "links": self.getLinks(),
            }
        elif messagetype == Message.Type.PROPAGATIONDATA:
            return {
            "parent_type": 'LinkEnvironmentGroundToSpace',
            "object_type": self.__class__.__name__,
            "id": self._id,
            "links": self.getLinks(),
            }
        else:
            raise ValueError("Not implemented yet.")
    
    async def _timestep_generic(self, scenario):
        if not self._initialized:
            self._time_start = scenario.getSimulationTime() # save data for future calculation
            self._observe(scenario=scenario, timediffsec=None, start=True)
        else:
            self._time_end = scenario.getSimulationTime() # save data for current calculation
            self._stepsize_sec = (self._time_end - self._time_start).total_seconds()
            self._observe(scenario=scenario, timediffsec=self._stepsize_sec, start=False)  # cache object pos+vel   
            for link in self._links:
                link._IQTimestep_prepare(scenario = scenario)

    async def _timestep_iq(self, scenario):
        if not self._initialized:
            self._initialized = True
        else:
            for link in self._links:
                await link._IQTimestep_iq(scenario = scenario, stepsize_sec = self._stepsize_sec)
            self._switch()


    def _observe(self, scenario, timediffsec: float, start: Optional[bool] = False):
        #if getVelocity doesnt exist (e.g. for EarthStationary) calculate velocity from 2 sequential positions manually.
        if start:
            if hasattr(self._groundObject, 'getVelocity'):
                self._groundObject_start = self._nd([self._groundObject.getPosition(), self._groundObject.getVelocity()])
            else:
                self._groundObject_start = self._nd([self._groundObject.getPosition(), [np.nan,np.nan,np.nan]])
            
            if hasattr(self._spaceObject, 'getVelocity'):
                self._spaceObject_start = self._nd([self._spaceObject.getPosition(), self._spaceObject.getVelocity()])
            else:
                self._spaceObject_start = self._nd([self._spaceObject.getPosition(), [np.nan,np.nan,np.nan]])

        else:
            if hasattr(self._groundObject, 'getVelocity'):
                self._groundObject_end = self._nd([self._groundObject.getPosition(), self._groundObject.getVelocity()])
            else:
                velocity = (self._groundObject.getPosition() - self._groundObject_start[0])/timediffsec
                self._groundObject_end = self._nd([self._groundObject.getPosition(), velocity])
                if self._groundObject_start[1][0] == np.nan:
                    self._groundObject_start[1] = velocity
            
            if hasattr(self._spaceObject, 'getVelocity'):
                self._spaceObject_end = self._nd([self._spaceObject.getPosition(), self._spaceObject.getVelocity()])
            else:
                velocity = (self._spaceObject.getPosition() - self._spaceObject_start[0])/timediffsec
                self._spaceObject_end = self._nd([self._spaceObject.getPosition(), velocity])
                if self._spaceObject_start[1][0] == np.nan:
                    self._spaceObject_start[1] = velocity

    def _switch(self,):
        self._time_start = self._time_end
        self._groundObject_start = self._groundObject_end
        self._spaceObject_start = self._spaceObject_end
    
    def _nd(self, list: List) -> np.ndarray:
        return np.array(list, dtype=np.float64)



class LinkEnvironmentGroundGroupToSpaceGroup(RequireEncode):
    r"""Implementation of the high-level link environment object for multiple ground-to-space links.
    This class is internally used by the RFEmulator object. The instantiated object holds information
    that is shared between all low-level links that are subcomponents of this link environment.

    Args:
    - rfemulator: RFEmulator object.
    - groundObjects: Either a single Stationary object or a list of Stationary objects.
    - spaceGroup: Collection of space objects of 'PropagatableGroup' type.

    """
    def __init__(self, rfemulator: RFEmulator, groundObjects: Union['Stationary', List['Stationary']], spaceGroup: 'PropagatableGroup'):
        self._rfemulator = rfemulator
        self._id = randUUID()
        self._groundObjects = groundObjects if isinstance(groundObjects, List) else [groundObjects] # enforce List
        self._spaceGroup = spaceGroup
        self._links = []
        # operating parameters
        self._initialized = False
        self._time_start = None # start of timeframe
        self._groundObjects_start = {} # GroundObject pos+vel at previous timestep
        self._groundObjects_end = {} # GroundObject pos+vel at current timestep
        self._time_end = None   # end of timeframe
        self._spaceObjects_start = {} # SpaceObject pos+vel at previous timestep
        self._spaceObjects_end = {}   # SpaceObject pos+vel at current timestep
        self._channel_time = 0 # keeping track of channel time
        
        for stationary in self.getGroundObjects():
            assert stationary.getRFFrontend() != None and len(stationary.getRFFrontend().getAntennas()) > 0 and len(stationary.getRFFrontend().getAntennas()[0].getBeams()) > 0, "All stationaries must have their RF Frontend with at least 1 antenna and 1 beam defined"
        for propagatable in self.getSpaceGroup().getComponents():
            assert propagatable.getRFFrontend() != None and len(propagatable.getRFFrontend().getAntennas()) > 0 and len(propagatable.getRFFrontend().getAntennas()[0].getBeams()) > 0, "All propagatables must have their RF Frontend with at least 1 antenna and 1 beam defined"

    class Link(RequireEncode, LinkBase):
        r"""Implementation of the low-level link object for a singular ground-to-space link within a group-based
        link environment. This class is internally used by the LinkEnvironmentGroundGroupToSpaceGroup object 
        and is optimized for efficient data access and handling of multiple links.
        
        Args:
        - environment: Specifies the Parent LinkEnvironmentGroundGroupToSpaceGroup object
        - direction: Specifies the link direction (downlink/uplink)
        - groundObject: Specifies the ground object within the group
        - spaceObject: Specifies the space object within the group
        - groundBeam: Specifies the ground beam object
        - spaceBeam: Specifies the space beam object
        - centerFrequency: Specifies the center frequency of the link
        - shadowScenario: Specifies the shadow scenario
        - minElevation: Specifies the minimum elevation angle
        - visualizeLinkStats: Specifies if link stats should be visualized in the frontend
        - label: Specifies the label of the link to be displayed in the frontend

        """
                
        def __init__(self, environment: 'LinkEnvironmentGroundGroupToSpaceGroup', 
                     direction: 'LinkDirection', 
                     groundObject: 'Stationary', 
                     spaceObject: 'Propagatable', 
                     groundBeam: RFFrontend.Antenna.Beam, 
                     spaceBeam: RFFrontend.Antenna.Beam, 
                     centerFrequency: float, 
                     shadowScenario: Optional[str]=None,
                     angleGSO: Optional[float] = 5,
                     minElevation: Optional[float]=5,
                     visualizeLinkStats: Optional[bool]=True,
                     label: Optional[str]=None):
            self._id = randUUID()
            self._environment = environment
            self._direction = direction
            self._emulateIQ = False
            self._groundObject = groundObject
            self._spaceObject = spaceObject
            self._groundBeam = groundBeam
            self._spaceBeam = spaceBeam
            self._centerFrequency = centerFrequency
            self._shadowScenario = shadowScenario
            self._angleGSO = angleGSO
            self._minElevation = minElevation
            self._label = label
            self._visualizeLinkStats = visualizeLinkStats
            self._linklevel = LinkLevelGroundToSpace(link = self)

        def getEnvironment(self,) -> 'LinkEnvironmentGroundGroupToSpaceGroup':
            r"""Return the parent LinkEnvironmentGroundGroupToSpaceGroup object.
            
            Returns:
                LinkEnvironmentGroundGroupToSpaceGroup: Parent object.
            """
            return self._environment
        
        def getLinkDirection(self,) -> LinkDirection:
            r"""Return the link direction.
            
            Returns:
                LinkDirection: Link direction.
            """
            return self._direction
        
        def getGroundObject(self,) -> 'Stationary':
            r"""Return the ground object.
            
            Returns:
                Stationary: Ground object.
            """
            # Todo: Note: Not sure if this utility function is needed.
            return self._groundObject
        
        def getSpaceObject(self,) -> 'Propagatable':
            r"""Return the space object.
            
            Returns:
                Propagatable: Space object.
            """
            # Todo: Note: Not sure if this utility function is needed.
            return self._spaceObject
        
        def getGroundBeam(self,) -> RFFrontend.Antenna.Beam:
            r"""Return the ground beam object.
            
            Returns:
                RFFrontend.Antenna.Beam: Ground beam object.
            """
            return self._groundBeam
        
        def getSpaceBeam(self,) -> RFFrontend.Antenna.Beam:
            r"""Return the space beam object.
            
            Returns:
                RFFrontend.Antenna.Beam: Space beam object.
            """
            return self._spaceBeam
        
        def getCenterFrequency(self,) -> float:
            r"""Return the center frequency.
            
            Returns:
                float: Center frequency.
            """
            return self._centerFrequency

        def getShadowScenario(self,) -> Optional[str]:
            r"""Return the shadow scenario.
            
            Returns:
                Optional[str]: Shadow scenario type as string.
            """
            return self._shadowScenario
        
        def getMinElev(self,) -> float:
            r"""Return the minimum elevation angle.
            
            Returns:
                float: Minimum elevation angle.
            """
            return self._minElevation

        def _getSource(self,) -> RFSource:
            r"""Return the RFSource object.
            
            Returns:
                RFSource: RFSource object associated with the link's IQ emulation.
            """
            assert self._emulateIQ == True, 'Function only available if IQ emulation activated'
            return self._IQRfsource

        def _getSink(self, ) -> RFSink:
            r"""Return the RFSink object.
            
            Returns:
                RFSink: RFSink object associated with the link's IQ emulation.
            """
            assert self._emulateIQ == True, 'Function only available if IQ emulation activated'
            return self._IQRfsink
        
        def getSampleRate(self,) -> int:
            r"""Return the sample rate of the link.
            
            Returns:
                int: Sample rate.
            """
            assert self._emulateIQ == True, 'Function only available if IQ emulation activated'
            return self._IQSampleRate
        
        def getTiming(self,) -> RealTiming:
            r"""Return the timing object of the link. Used for internal timing and performance optimization purposes.
            
            Returns:
                RealTiming: Timing object.
            """
            assert self._emulateIQ == True, 'Function only available if IQ emulation activated'
            return self._IQTiming
        
        def emulateIQ(self, sampleRate: int, rfsourcefct: Callable, rfsinkfct: Callable, sinkBatchSize: Optional[int] = None, bufferSize: Optional[int]=5, visualize: Optional[bool] = False, onProcessed : Optional[bool] = None):
            r"""Activate IQ emulation for the link with the specified parameters
            
            Args:
            - sampleRate: Sample rate in Hz
            - rfsourcefct: Function that generates IQ samples
            - rfsinkfct: Function that processes IQ samples
            - sinkBatchSize: Batch size of the sink
            - bufferSize: Multiplier for calculating the buffer size of the internal IQ buffers
            - visualize: If True, visualize the IQ data in the frontend
            - onProcessed: Callable that will be invoked when a IQ emulation step is concluded

            """
            self._emulateIQ = True
            self._IQSampleRate = sampleRate
            self._IQRfsource = RFSource(inputfct=rfsourcefct)
            self._IQRfsink = RFSink(outputfct=rfsinkfct)
            self._IQsinkBatchSize = sinkBatchSize
            self._IQBufferSize = bufferSize
            self._IQvisualize = visualize
            self._IQvisualize_type = RFEmulator.LinkData.SPECTROGRAM if visualize else None
            self._IQOnProcessed = onProcessed
            self._IQTiming = RealTiming()
        
        def linklevel(self,) -> LinkLevelGroundToSpace:
            r"""Return the used link level object.
            
            Returns:
                LinkLevelGroundToSpace: Link level object used for defining link characteristics.
            """
            return self._linklevel
            
        def _initialize(self, scenario):
            timestep_secs = scenario.getTimestep().total_seconds()
            if self._emulateIQ:
                self._IQsinkBatchSize = self._IQsinkBatchSize if self._IQsinkBatchSize != None else int(self._IQSampleRate*timestep_secs) # one output per timestep
                self._IQRfsource._initBuffer(buffersize=int(self._IQSampleRate*timestep_secs*self._IQBufferSize))
                self._IQRfsink._initBuffer(buffersize=int(self._IQSampleRate*timestep_secs*self._IQBufferSize), outputbatchsize=self._IQsinkBatchSize)
                self._IQRfsource._start()
                self._IQRfsink._start()

        def _IQTimestep_prepare(self, scenario,):  
            if self._emulateIQ or self._visualizeLinkStats:
                # compute link level stats
                self._lastlinkleveldata = self.linklevel().calculateLinkBudget(returnAll=True)
                if self._emulateIQ:
                    self._lastlinkleveldata["processingrateiq"] = self._IQTiming.processedThroughput()

               
        
        async def _IQTimestep_iq(self, scenario, stepsize_sec: float):
            # IQ data has been processed. Prepare for visualization as specific in self._IQvisualize_type and send to website:
            if self._emulateIQ:
                numsamps = np.array(np.round(self._IQSampleRate*stepsize_sec), dtype=np.int64) # todo fix for fractional # calculate number of samples to be processed in current timeframe
                iq_source = self._IQRfsource._get(numsamples=numsamps)
                iq_processed = self._processIQ(iq = iq_source)
                self._IQRfsink._push(iq=iq_processed)
                self._IQTiming.count(numsamps)
                if self._IQOnProcessed != None:
                    self._IQOnProcessed(self)
                self.iq_processed = iq_processed
                self.numsamps = numsamps
                if self._environment._rfemulator._visualization and self._emulateIQ and self._IQvisualize_type == RFEmulator.LinkData.SPECTROGRAM:
                    iq_processed = self.iq_processed
                    numsamps = self.numsamps
                    # calculate spectrum with fft (no windowing)
                    tf = getPyModule("tensorflow") if RFEMULATOR_GPU_IQ_OFFLOAD_TENSORFLOW else None                   
                    fft_max = 256
                    fft_len = fft_max if numsamps > fft_max else numsamps
                    #truncate from center
                    trunc_start = int(numsamps/2 - fft_len/2)
                    #fft = tf.signal.fft(iq_processed[:fft_len]) if RFEMULATOR_GPU_IQ_OFFLOAD_TENSORFLOW else np.fft.fft(iq_processed[:fft_len])                 
                    fft = np.fft.fft(iq_processed[trunc_start:int(trunc_start+fft_len)])
                    fft = fft if RFEMULATOR_GPU_IQ_OFFLOAD_TENSORFLOW else np.array(fft, dtype=np.complex64)
                    fft = tf.signal.fftshift(fft) if RFEMULATOR_GPU_IQ_OFFLOAD_TENSORFLOW else np.fft.fftshift(fft)
                    magnitude = tf.abs(fft) if RFEMULATOR_GPU_IQ_OFFLOAD_TENSORFLOW else np.abs(fft)
                    power = magnitude**2
                    power_mW = power * 1000
                    power_dBm = (10 * tf.math.log(power_mW, 10) / tf.math.log(10.0)).numpy() if RFEMULATOR_GPU_IQ_OFFLOAD_TENSORFLOW else 10 * np.log10(power_mW) #dtype=float32
                    await self._environment._rfemulator._sendLinkData(link=self, data=power_dBm)
                else:
                    pass
        
        def _processIQ(self, iq) -> np.ndarray:
            # Fetch positions and velocities
            space_object = self.getSpaceObject()
            ground_object = self.getGroundObject()
            groundInfo = [self.getEnvironment()._groundObjects_start[ground_object.getId()], self.getEnvironment()._groundObjects_end[ground_object.getId()]]
            spaceInfo = [self.getEnvironment()._spaceObjects_start[space_object.getId()], self.getEnvironment()._spaceObjects_end[space_object.getId()]]

            #processedIQ_1 = applyDopplerIQEarthSpace(link=self, groundInfo=groundInfo, spaceInfo=spaceInfo, iq=iq)
            processedIQ_2 = applyDopplerIQEarthSpaceEfficient(link=self, groundInfo=groundInfo, spaceInfo=spaceInfo, iq=iq)
            #error = np.sum(processedIQ_1-processedIQ_2)
            #print("Error: ", error)
            return processedIQ_2    

        def encode(self, messagetype: Message.Type) -> Dict:
            # Internal method to encode the object for message passing
            if messagetype == Message.Type.SCENARIOSETUP:
                return {
                "parent_type": 'LinkEnvironmentGroundGroupToSpaceGroup.Link',
                "object_type": self.__class__.__name__,
                "id": self._id,
                "label": self._label,
                "groundObject": self._groundObject.getId(),
                "spaceObject": self._spaceObject.getId(),
                "groundAntennaIdx": self._groundBeam.getAntenna()._antIdx,
                "groundBeamIdx": self._groundBeam._beamIdx,
                "spaceAntennaIdx": self._spaceBeam.getAntenna()._antIdx,
                "spaceBeamIdx": self._spaceBeam._beamIdx,
                "linkdirection": self._direction.name,
                "centerFrequency": self._centerFrequency,
                "minElevation": self._minElevation,
                "iqstream": self._emulateIQ,
                **({"iq_samplerate": self._IQSampleRate} if self._emulateIQ else {}),
                **({"iq_visualize": self._IQvisualize} if self._emulateIQ else {}),
                "visualizeLinkStats": self._visualizeLinkStats,
                }
            elif messagetype == Message.Type.PROPAGATIONDATA:
                return {
                "id": self._id,
                **({"lastlinkleveldata": self._lastlinkleveldata} if hasattr(self, "_lastlinkleveldata") else {}),
                }
            else:
                raise ValueError("Not implemented yet.")

    
    def getLinks(self,) -> List['Link']:
        r"""Returns all registered links.
        
        Returns:
            List[Link]: List of all registered links.
        """
        return self._links

    def getGroundObjects(self,) -> List['Stationary']:
        r"""Returns the ground objects.
        
        Returns:
            List[Stationary]: List of all ground objects.
        """
        return self._groundObjects
    
    def getSpaceGroup(self,) -> 'PropagatableGroup':
        r"""Returns the space group.
        
        Returns:
            PropagatableGroup: Space group object.
        """
        return self._spaceGroup

    def _initialize(self, scenario):
        for link in self._links:
            link._initialize(scenario=scenario)

    def registerAllCrossLinks(self, direction: LinkDirection, 
                           beamMapper: Optional[Union[Callable, tuple]]=None, # for now I will add a list as a beammapper
                           centerFrequency: Optional[Union[float, Callable]] = None,
                           shadowScenario: Optional[str]=None,
                           minElevation: Optional[float]=5,
                           visualizeLinkStats: Optional[bool]=True) -> List[Link]:
        r"""Registers all cross links between the ground objects and the space objects.
        
        Args:
        - direction: Specifies the link direction (downlink/uplink)
        - beamMapper: Specifies the beam mapping between the ground and space objects. If None, all beams are used.
        - centerFrequency: Specifies the center frequency of the link
        - shadowScenario: Specifies the shadow scenario
        - minElevation: Specifies the minimum elevation angle
        - visualizeLinkStats: Specifies if link stats should be visualized in the frontend

        Returns:
            List[Link]: List of all registered links.
        """
        # Todo later - Michael: Implement various function arguments.... for now, stick with simplest antenna+beam mapping 0/0 <-> 0/0
        links = []
        
        # either assign all beams or differentiate based on link direction? Would logically also make sense
        if beamMapper is None:
            for stationary in self.getGroundObjects():           
                groundBeam = stationary.getRFFrontend().getCertainAntenna(0).getCertainBeam(0)
                for propagatable in self.getSpaceGroup().getComponents():
                    spaceBeam = propagatable.getRFFrontend().getCertainAntenna(0).getCertainBeam(0)
                    link = LinkEnvironmentGroundGroupToSpaceGroup.Link(environment = self, 
                                                                       direction = direction, 
                                                                       groundObject = stationary, 
                                                                       spaceObject = propagatable, 
                                                                       groundBeam = groundBeam, 
                                                                       spaceBeam = spaceBeam, 
                                                                       centerFrequency = centerFrequency,
                                                                       shadowScenario = shadowScenario,
                                                                       minElevation = minElevation,
                                                                       visualizeLinkStats = visualizeLinkStats)
                    links.append(link)
                    self._links.append(link)
        else:
            for i in range(0,len(beamMapper), 2):
                for stationary in self.getGroundObjects():           
                    groundBeam = stationary.getRFFrontend().getCertainAntenna(0).getCertainBeam(beamMapper[i])
                    for propagatable in self.getSpaceGroup().getComponents():
                        spaceBeam = propagatable.getRFFrontend().getCertainAntenna(0).getCertainBeam(beamMapper[i+1])
                        link = LinkEnvironmentGroundGroupToSpaceGroup.Link(environment = self, 
                                                                           direction = direction, 
                                                                           groundObject = stationary, 
                                                                           spaceObject = propagatable, 
                                                                           groundBeam = groundBeam, 
                                                                           spaceBeam = spaceBeam, 
                                                                           centerFrequency = centerFrequency,
                                                                           shadowScenario = shadowScenario,
                                                                           minElevation = minElevation,
                                                                           visualizeLinkStats = visualizeLinkStats)
                        links.append(link)
                        self._links.append(link)

        return links

    def encode(self, messagetype: Message.Type) -> Dict:   
        # Internal method for encoding the Linkenvironment object for message passing     
        if messagetype == Message.Type.SCENARIOSETUP:
            return {
            "parent_type": 'LinkEnvironmentGroundGroupToSpaceGroup',
            "object_type": self.__class__.__name__,
            "id": self._id,
            "groundObjects": [groundObject.getId() for groundObject in self._groundObjects],
            "spaceGroup": self._spaceGroup.getId(),
            "links": self.getLinks(),
            }
        elif messagetype == Message.Type.PROPAGATIONDATA:
            return {
            "parent_type": 'LinkEnvironmentGroundGroupToSpaceGroup',
            "object_type": self.__class__.__name__,
            "id": self._id,
            "links": self.getLinks(),
            }
        else:
            raise ValueError("Not implemented yet.")
    
    async def _timestep_generic(self, scenario):
        if not self._initialized:
            self._time_start = scenario.getSimulationTime() # save data for future calculation
            self._observe(scenario=scenario, timediffsec=None, initialized=False)
        else:
            self._time_end = scenario.getSimulationTime() # save data for current calculation
            self._stepsize_sec = (self._time_end - self._time_start).total_seconds()
            self._observe(scenario=scenario, timediffsec=self._stepsize_sec, initialized=True)  # cache object pos+vel  
            for link in self._links:
                link._IQTimestep_prepare(scenario = scenario)
    
    
    async def _timestep_iq(self, scenario):
        if not self._initialized:
            self._initialized = True
        else:
            for link in self._links:
                await link._IQTimestep_iq(scenario = scenario, stepsize_sec = self._stepsize_sec)
            self._switch()

    def _observe(self, scenario: 'Scenario', timediffsec: float, initialized: bool):
        for stationary in self.getGroundObjects():
            if not initialized:
                if hasattr(stationary, 'getVelocity'):
                    #if getVelocity doesnt exist (e.g. for EarthStationary) calculate velocity from 2 sequential positions manually.
                    self._groundObjects_start[stationary.getId()] = self._nd([stationary.getPosition(), stationary.getVelocity()])
                else:
                    self._groundObjects_start[stationary.getId()] = self._nd([stationary.getPosition(), [np.nan,np.nan,np.nan]])
            else:
                if hasattr(stationary, 'getVelocity'):
                    self._groundObjects_end[stationary.getId()] = self._nd([stationary.getPosition(), stationary.getVelocity()])
                else:
                    velocity = (stationary.getPosition() - self._groundObjects_start[stationary.getId()][0])/timediffsec
                    self._groundObjects_end[stationary.getId()] = self._nd([stationary.getPosition(), velocity])
                    if self._groundObjects_start[stationary.getId()][1][0] == np.nan:
                        self._groundObjects_start[stationary.getId()][1] = velocity # set start velocity if start has none, so that both start and end velocities are initialized.

        for propagatable in self.getSpaceGroup().getComponents():
            if not initialized:
                if hasattr(propagatable, 'getVelocity'):
                    self._spaceObjects_start[propagatable.getId()] = self._nd([propagatable.getPosition(), propagatable.getVelocity()])
                else:
                    self._spaceObjects_start[propagatable.getId()] = self._nd([propagatable.getPosition(), [np.nan,np.nan,np.nan]])
            else:
                if hasattr(propagatable, 'getVelocity'):
                    self._spaceObjects_end[propagatable.getId()] = self._nd([propagatable.getPosition(), propagatable.getVelocity()])
                else:
                    velocity = (propagatable.getPosition() - self._spaceObjects_start[propagatable.getId()][0])/timediffsec
                    self._spaceObjects_end[propagatable.getId()] = self._nd([propagatable.getPosition(), velocity])
                    if self._spaceObjects_start[propagatable.getId()][1][0] == np.nan:
                        self._spaceObjects_start[propagatable.getId()][1] = velocity # set start velocity if start has none, so that both start and end velocities are initialized.

    def _switch(self,):
        self._time_start = self._time_end
        self._groundObjects_start = self._groundObjects_end
        self._spaceObjects_start = self._spaceObjects_end
    
    def _nd(self, list: List) -> np.ndarray:
        return np.array(list, dtype=np.float64)


def applyDopplerIQEarthSpaceEfficient(link: LinkBase, groundInfo, spaceInfo, iq, print1=False) -> np.ndarray:
    r"""CPU implementation with numpy for efficiently applying a sample-wise calculated Doppler shift to the IQ data
    for the moving scenario. The method calculates the Doppler shift based on the relative velocity along the line-of-sight
    direction between the ground and space objects. A elevation angle is also calculated and IQ samples that are when the
    minimum elevation angle is not sustained are discarded. The method returns the IQ data with the Doppler shift applied.
    
    Args:
    - link: Link object
    - groundInfo: Ground object position and velocity vectors
    - spaceInfo: Space object position and velocity vectors
    - iq: IQ data to apply Doppler shift to
    - print1: If True, print debug information

    Returns:
        np.ndarray: IQ data with Doppler shift applied
    
    """
    # Internal method for applying Doppler shift to IQ data
    space_object = link.getSpaceObject()
    
    # Extract position and velocity vectors correctly
    pos_gnd_start = np.array(groundInfo[0][0])  # unit [km]
    pos_gnd_end = np.array(groundInfo[1][0])    # unit [km]
    v_gnd_start = np.array(groundInfo[0][1])    # unit [km/s]
    v_gnd_end = np.array(groundInfo[1][1])      # unit [km/s]
    pos_sat_start = np.array(spaceInfo[0][0])   # unit [km]
    pos_sat_end = np.array(spaceInfo[1][0])     # unit [km]
    v_sat_start = np.array(spaceInfo[0][1])     # unit [km/s]
    v_sat_end = np.array(spaceInfo[1][1])       # unit [km/s]
    
    # Time interpolation
    environment = link._environment
    start_time_rel = 0
    end_time_rel = (environment._time_end - environment._time_start).total_seconds()
    sample_size = len(iq)

    # Calculate relative velocity (start and end)
    vrel_start = v_gnd_start - v_sat_start
    vrel_end = v_gnd_end - v_sat_end

    # Calculate line-of-sight direction from sat to gnd (start and end)
    dir_start = pos_gnd_start - pos_sat_start
    dir_end = pos_gnd_end - pos_sat_end
    dir_unit_start = dir_start / np.linalg.norm(dir_start)
    dir_unit_end = dir_end / np.linalg.norm(dir_end)

    # Calculate elevation angles (start and end)
    dirVec_start = -dir_start
    nom_start = np.sum(pos_gnd_start * dirVec_start)
    den_start = np.sqrt(np.sum(pos_gnd_start**2) * np.sum(dirVec_start**2))
    el_start = np.arccos(nom_start / den_start) * 180 / np.pi
    elevation_angle_rad_start = (90 - el_start) / 180 * np.pi

    dirVec_end = -dir_end
    nom_end = np.sum(pos_gnd_end * dirVec_end)
    den_end = np.sqrt(np.sum(pos_gnd_end**2) * np.sum(dirVec_end**2))
    el_end = np.arccos(nom_end / den_end) * 180 / np.pi
    elevation_angle_rad_end = (90 - el_end) / 180 * np.pi

    # Calculate relative velocity along LoS components (start and end)
    vrel_los_start = np.sum(vrel_start * dir_unit_start)  # Dot product, scalar
    vrel_los_end = np.sum(vrel_end * dir_unit_end)        # Dot product, scalar

    # Calculate doppler shift (start and end)
    f_c = link.getCenterFrequency()
    f_d_start = -f_c * (vrel_los_start / 3e5)  # Doppler shift [Hz]
    f_d_end = -f_c * (vrel_los_end / 3e5)      # Doppler shift [Hz]

    initial_phase = link._rfemulator_initial_phase if hasattr(link, "_rfemulator_initial_phase") else np.array(0, np.float32)  # Load "end" phase as initial phase

    # Interpolate data    
    t = np.linspace(start_time_rel, end_time_rel, sample_size+1, endpoint=True)  # Including endpoint for phase continuity
    t_norm = t / end_time_rel  # Normalized time vector
    interp_elevangle = elevation_angle_rad_start * (1 - t_norm) + elevation_angle_rad_end * t_norm
    interp_fd = f_d_start * (1 - t_norm) + f_d_end * t_norm

    # Remove iq samples if below min-elev
    iq_discard_idx = interp_elevangle < (link.getMinElev() / 180 * np.pi)
    iq[iq_discard_idx[:-1]] = 0  # Remove signal where elevation angle < min-elev

    # Apply doppler shift
    phase_doppler = 2 * np.pi * interp_fd * t + initial_phase
    iq_shifted = iq * np.exp(1j * phase_doppler[:-1])  # Add initial phase as offset, ensuring phase continuity
    
    link._rfemulator_initial_phase = phase_doppler[-1] if not np.isnan(phase_doppler[-1]) else 0  # Save last phase into (next) initial phase
    if np.isnan(phase_doppler[0]):
        print("Error calculating doppler for ", space_object.getName(), ". Check positions and velocities of involved objects.")

    if print1: #Debug
        print("Efficient method:")
        print(space_object.getName())
        print("intpdoppler start/end", interp_fd[0], interp_fd[-1])
        print("intp.elevation start/end", interp_elevangle[0]*180/np.pi, interp_elevangle[-1]*180/np.pi)
        print("doppler start/end", f_d_start, f_d_end)
        print("elevation start/end", elevation_angle_rad_start*180/np.pi, elevation_angle_rad_end*180/np.pi)
        print("phase_doppler start/end", phase_doppler[0], phase_doppler[-1])
        print("groundInfo", groundInfo)
        print("spaceInfo", spaceInfo)
        print("end_time_rel", end_time_rel)

    return iq_shifted
