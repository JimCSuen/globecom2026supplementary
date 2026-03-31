#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: constellation.py
Author: Li ZENG @ HKUST ECE
License: MIT License

Description:
This module defines the WalkerConstellation abstract base class and its derived classes for managing satellite constellations.
It includes methods for initializing constellations, updating satellite positions, and calculating phase differences.
"""

from abc import ABC, abstractmethod
import numpy as np
from .orbit import Orbit, ServiceOrbit

EARTH_RADIUS = 6371  # Define Earth's radius as a constant


class WalkerConstellation(ABC):
    """
    Abstract base class representing a satellite constellation.
    """

    def __init__(self, num_orbits, num_sats_per_orbit, radius):
        """
        Initialize the constellation's properties.

        Parameters:
        - num_orbits: Number of orbital planes.
        - num_sats_per_orbit: Number of satellites per orbital plane.
        - radius: Radius of the orbits in km.
        """
        self.time = 0  # Simulation time in seconds
        self.radius = radius  # Radius of the orbits in km
        self.altitude = (
            radius - EARTH_RADIUS
        )  # Altitude of the orbits in km (Earth's radius is EART_RADIUS = 6371 km)
        self.num_orbits = num_orbits  # Number of orbital planes
        self.num_sats_per_orbit = (
            num_sats_per_orbit  # Number of satellites per orbital plane
        )

    def update_constellation(self, delta_t):
        """
        Update the positions of all satellites in the constellation based on the time increment.

        Parameters:
        - delta_t: Time increment in seconds.
        """
        self.time += delta_t
        for orbit in self.orbits:
            orbit.update_orbit(delta_t)


class StarConstellation(WalkerConstellation):
    """
    Class representing a Walker Star Constellation.
    """

    def __init__(
        self, num_orbits=12, num_sats_per_orbit=30, radius=EARTH_RADIUS + 550.0
    ):
        """
        Initialize the Walker Star Constellation.

        Parameters:
        - num_orbits: Number of orbital planes (default: 12).
        - num_sats_per_orbit: Number of satellites per orbital plane (default: 30).
        - radius: Radius of the orbits in km (default: EARTH_RADIUS + 550).
        """
        super().__init__(num_orbits, num_sats_per_orbit, radius)
        self.type = "Walker Star Constellation"
        self.inclination = np.radians(90)  # Inclination of 90 degrees for polar orbits
        self.phasediff = 0  # No phase difference for Walker Star Constellation
        self.orbits = [
            Orbit(
                self, orbit_id, self.radius, self.inclination, self.num_sats_per_orbit
            )
            for orbit_id in range(1, num_orbits + 1)
        ]


class DeltaConstellation(WalkerConstellation):
    """
    Class representing a Walker Delta Constellation.
    """

    def __init__(
        self,
        num_orbits=12,
        num_sats_per_orbit=30,
        radius=EARTH_RADIUS + 550.0,
        inclination=53.0,
        phasediff=0,
    ):
        """
        Initialize the Walker Delta Constellation.

        Parameters:
        - num_orbits: Number of orbital planes (default: 12).
        - num_sats_per_orbit: Number of satellites per orbital plane (default: 30).
        - radius: Radius of the orbits in km (default: EART_RADIUS + 550.0).
        - inclination: Inclination of the orbits in degrees (default: 53.0).
        """
        if inclination <= 0 or inclination >= 90:
            raise ValueError("Inclination must be between 0 and 90 degrees.")

        super().__init__(num_orbits, num_sats_per_orbit, radius)
        self.type = "Walker Delta Constellation"
        self.inclination = np.radians(inclination)  # Convert inclination to radians
        self.phasediff = phasediff  # Calculate phase difference
        self.orbits = [
            Orbit(
                self, orbit_id, self.radius, self.inclination, self.num_sats_per_orbit
            )
            for orbit_id in range(1, num_orbits + 1)
        ]


class ServiceDeltaConstellation(DeltaConstellation):
    """
    Class representing a Walker Delta Constellation.
    """

    def __init__(
        self,
        num_orbits=12,
        num_sats_per_orbit=30,
        radius=EARTH_RADIUS + 550.0,
        inclination=53.0,
        phasediff=0,
        fs=1000,
    ):
        """
        Initialize the Walker Delta Constellation.

        Parameters:
        - num_orbits: Number of orbital planes (default: 12).
        - num_sats_per_orbit: Number of satellites per orbital plane (default: 30).
        - radius: Radius of the orbits in km (default: EART_RADIUS + 550.0).
        - inclination: Inclination of the orbits in degrees (default: 53.0).
        """
        if inclination <= 0 or inclination >= 90:
            raise ValueError("Inclination must be between 0 and 90 degrees.")

        super().__init__(num_orbits, num_sats_per_orbit, radius)
        self.type = "Walker Delta Constellation"
        self.num_orbits = int(num_orbits)
        self.N_const = num_orbits
        self.num_sats_per_orbit = int(num_sats_per_orbit)
        self.num_sats = int(num_orbits * num_sats_per_orbit)
        self.inclination = np.radians(inclination)  # Convert inclination to radians
        self.phasediff = phasediff  # Calculate phase difference
        self.num_isls = 0

        self.orbits = [
            ServiceOrbit(
                self,
                orbit_id,
                self.radius,
                self.inclination,
                self.num_sats_per_orbit,
                fs,
            )
            for orbit_id in range(1, num_orbits + 1)
        ]
        self.fs = fs
