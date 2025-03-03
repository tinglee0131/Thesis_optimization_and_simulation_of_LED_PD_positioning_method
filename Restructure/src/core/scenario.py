import numpy as np
from typing import Optional


class TestPoint:
    """Class for managing test points in the positioning system."""
    
    def __init__(self, scenario: int, extra_params: Optional[dict] = None):
        """
        Initialize test points based on scenario.
        
        Args:
            scenario: Scenario number
                0: 3D space with translation and rotation
                1: 2D plane
                2: Spherical space
                3: Extended version of 0
                4: 1-to-1 specific test point
            extra_params: Additional parameters for the scenario
                For scenario 3, this should include 'ma' (space size)
        """
        
        if scenario == 0:
            # Translation samples in 3D space
            self.testp_pos = np.mgrid[-1.5:1.5:10j, -1.5:1.5:10j, 0:3:10j].reshape((3, -1))
            
            # Rotation samples
            self.testp_rot = ((np.mgrid[0:0:1j, 10:60:6j, 0:360:11j])[:, :, :, :-1])
            self.testp_rot = np.deg2rad(self.testp_rot.reshape((3, -1)))
            self.testp_rot = np.concatenate((self.testp_rot, np.array([[0, 0, 0]]).T), axis=1) + np.array([[np.pi, 0, 0]]).T
        
        elif scenario == 1:
            # 2D plane
            self.testp_pos = np.mgrid[-1.5:1.5:100j, -1.5:1.5:100j, 2.5:2.5:1j].reshape((3, -1))
            self.testp_rot = np.array([[np.pi, 0, 0]]).T
        
        elif scenario == 2:
            # Spherical space with arbitrary positions and orientations
            sample = 6
            # Distance samples in spherical coordinates
            dis_sample = np.linspace(0, 3, 4 + 1)[1:]
            
            # Zenith and azimuth angles for arbitrary orientation
            u, v = np.meshgrid(
                np.linspace(0, 2*np.pi, 2*sample + 1)[0:-1:1],
                np.linspace(0, np.pi, sample + 1)[1:-1:1]
            )
            
            # Add angles at the north pole
            u = np.append(u.reshape((-1,)), 0)
            v = np.append(v.reshape((-1,)), 0)
            
            # Add angles at the south pole
            u = np.append(u.reshape((-1,)), 0)
            v = np.append(v.reshape((-1,)), np.pi)
            
            # Convert spherical to Cartesian coordinates
            x = (1 * np.cos(u) * np.sin(v))
            y = (1 * np.sin(u) * np.sin(v))
            z = (1 * np.cos(v))
            
            # Combine x, y, z into a matrix [3 x ?]
            U = np.stack((x, y, z))
            
            # Consider different distances
            U = np.tile(U, (dis_sample.size, 1, 1)).transpose((1, 2, 0))
            self.testp_pos = np.multiply(dis_sample.reshape((1, 1, -1)), U)
            
            # Reshape to 2D [3 x ?]
            self.testp_pos = self.testp_pos.reshape((3, -1))
            
            # Rotation samples: pitch=0, roll and yaw with arbitrary orientation
            self.testp_rot = np.stack((np.zeros(u.shape), v, u))
        
        elif scenario == 3:
            # Extended version of scenario 0
            ma = extra_params["space_size"]  # Space size
            self.testp_pos = np.mgrid[-ma/2:ma/2:10j, -ma/2:ma/2:10j, 0:ma:10j].reshape((3, -1))
            self.testp_rot = ((np.mgrid[0:0:1j, 10:60:6j, 0:360:11j])[:, :, :, :-1])
            self.testp_rot = np.deg2rad(self.testp_rot.reshape((3, -1)))
            self.testp_rot = np.concatenate((self.testp_rot, np.array([[0, 0, 0]]).T), axis=1) + np.array([[np.pi, 0, 0]]).T
        
        elif scenario == 4:
            # 1-to-1 specific test point
            self.testp_pos = np.array([[1, 1, 1.5]]).T
            self.testp_rot = np.array([[np.pi, 0, 0]]).T
        
        # Store the number of position and rotation samples
        self.kpos = self.testp_pos.shape[1]
        self.krot = self.testp_rot.shape[1]

    
