# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: network.py
Author: Li ZENG @ HKUST ECE
License: MIT License

Description:
This module defines the satellite network classes for Walker Star and Walker Delta constellations.
It includes methods for building the network graph, updating satellite positions, and calculating shortest paths.
"""

from .constellation import (
    StarConstellation,
    DeltaConstellation,
    ServiceDeltaConstellation,
)
from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
import pyvista as pv
from pyvista import examples
import cvxpy as cp
import matplotlib.pyplot as plt

import time
import json

from pathlib import Path

from .result_record import result_record_class

# from NN_module import Predictor_NN, TaskAllocDataset
# import torch
# from torch import nn
# from torch.utils.data import DataLoader

import copy

EARTH_RADIUS = 6371.137  # Earth's radius, km
WINDOW_SIZE = (700, 500)  # figure size of the earth simulation


class SatNet(ABC):
    """
    Abstract base class for satellite networks.
    Provides common methods and properties for satellite networks.
    """

    __LIGHT_SPEED = 299792.458  # Speed of light in km/s

    @property
    def LIGHT_SPEED(self):
        """Speed of light in km/s (read-only)"""
        return self.__LIGHT_SPEED

    def __init__(self, constellation):
        """
        Initialize the satellite network.

        Parameters:
        - constellation: WalkerConstellation object representing the satellite constellation.
        """
        self.constellation = constellation
        self.time = 0  # Initial time in seconds

    def _build_graph(self):
        """Abstract method to build the network graph."""
        pass

    @abstractmethod
    def update_graph(self):
        """Abstract method to update the network graph."""
        pass

    def get_distance(self, vertex_key1, vertex_key2):
        """
        Calculate the distance between two satellites.

        Parameters:
        - vertex_key1: Tuple (orbit_id, sat_id) of the first satellite.
        - vertex_key2: Tuple (orbit_id, sat_id) of the second satellite.

        Returns:
        - Distance between the two satellites in km.
        """
        try:
            sat1 = self.graph.nodes[vertex_key1]["sat"]
            sat2 = self.graph.nodes[vertex_key2]["sat"]
        except KeyError:
            raise ValueError("One or both vertex keys do not exist in the network")
        pos1 = sat1.position_ecef
        pos2 = sat2.position_ecef
        return np.linalg.norm(pos1 - pos2)

    def update_network(self, delta_t):
        """
        Update the network by advancing the simulation time and updating satellite positions.

        Parameters:
        - delta_t: Time increment in seconds.
        """
        self.time += delta_t
        self.constellation.update_constellation(delta_t)
        self.update_graph()
        return

    def get_shortest_path(self, source, target, weight="weight"):
        """
        Find the shortest path between two satellites in the network.

        Parameters:
        - source: Tuple (orbit_id, sat_id) of the source satellite.
        - target: Tuple (orbit_id, sat_id) of the target satellite.
        - weight: Edge attribute to use as weight (default: 'weight').

        Returns:
        - path: List of vertex keys representing the shortest path.
        - latency: End-to-end propagation latency in seconds.
        """
        try:
            if source not in self.graph or target not in self.graph:
                raise ValueError("Source or target node not found in the network")

            # Find the shortest path using Dijkstra's algorithm
            path = nx.shortest_path(self.graph, source, target, weight="weight")

            # Calculate total distance along the path
            total_distance = 0
            for i in range(len(path) - 1):
                total_distance += self.graph[path[i]][path[i + 1]]["weight"]

            # Calculate latency (distance / speed of light)
            latency = total_distance / self.LIGHT_SPEED

            return path, latency

        except nx.NetworkXNoPath:
            raise ValueError(f"No path exists between satellites {source} and {target}")
        except Exception as e:
            raise Exception(f"Error finding shortest path: {str(e)}")

    def get_single_source_paths(self, source=(1, 1)):
        """
        Find shortest paths and latencies from a source satellite to all other satellites.

        Parameters:
        - source: Tuple (orbit_id, sat_id) of the source satellite.

        Returns:
        - paths: Dictionary {target: path} containing paths to all other satellites.
        - latencies: Dictionary {target: latency} containing latencies to all other satellites.
        """
        try:
            if source not in self.graph:
                raise ValueError("Source node not found in the network")

            # Get distances and paths to all nodes using single-source Dijkstra
            distances, paths = nx.single_source_dijkstra(
                self.graph, source, weight="weight"
            )

            # Convert distances to latencies
            latencies = {
                target: distance / self.LIGHT_SPEED
                for target, distance in distances.items()
            }

            return paths, latencies

        except Exception as e:
            raise Exception(f"Error computing single-source paths: {str(e)}")


class SatNetStar(SatNet):
    """
    Satellite network class for Walker Star Constellations.
    """

    def __init__(self, constellation: StarConstellation, gamma_deg=80.0):
        """
        Initialize the Walker Star satellite network.

        Parameters:
        - constellation: StarConstellation object representing the satellite constellation.
        - gamma_deg: Maximum latitude that supports the inter-plane links (default: 80.0 degrees).
        """
        if constellation.type != "Walker Star Constellation":
            raise ValueError("SatNetStar requires a Walker Star Constellation")

        super().__init__(constellation)
        self.gamma_deg = (
            gamma_deg  # Maximum latitude that supports the inter-plane links
        )
        self._build_graph()
        return

    def update_graph(self):
        """Update the network graph for the Walker Star Constellation."""
        self._build_graph()
        return

    def _build_graph(self):
        """
        Build the network graph for the Walker Star Constellation.

        Returns:
        - graph: NetworkX graph representing the satellite network.
        """
        graph = nx.Graph()

        # Add nodes for each satellite in the constellation
        for orbit in self.constellation.orbits:
            for sat in orbit.sats:
                # Add node with key as (orbit_id, satellite_id) and store the satellite object
                graph.add_node((orbit.id, sat.id), sat=sat)
        self.graph = graph

        # Add edges for laser inter-satellite links
        # * Intra-plane LISL: add edges between neighboring satellites in the same orbit
        for orbit in self.constellation.orbits:
            if orbit.num_sats < 2:
                continue  # No intra-plane links for single-satellite orbits
            for sat_id in range(1, orbit.num_sats + 1):
                sat1 = (orbit.id, sat_id)
                sat2 = (
                    orbit.id,
                    (sat_id % orbit.num_sats) + 1,
                )  # ! good! to form a circle loop
                if self._check_isl_feasibility(sat1, sat2):
                    distance = self.get_distance(sat1, sat2)

                    self.graph.add_edge(
                        sat1, sat2, weight=distance
                    )  # ! maybe change the attribute

        # * Inter-plane LISL: add edges between neighboring satellites in adjacent orbits
        if self.constellation.num_orbits < 2:
            return self.graph  # No inter-plane links for single-orbit constellations
        for orbit_id in range(
            1, self.constellation.num_orbits + 1
        ):  # just traverse, all the orbits, and sats
            for sat_id in range(1, self.constellation.num_sats_per_orbit + 1):
                sat1 = (orbit_id, sat_id)
                sat2 = ((orbit_id % self.constellation.num_orbits) + 1, sat_id)
                if self._check_isl_feasibility(sat1, sat2):
                    distance = self.get_distance(
                        sat1, sat2
                    )  # ! also, distance-based link establishment could be added
                    self.graph.add_edge(sat1, sat2, weight=distance)

        return self.graph  # ! important fault!

    def _check_isl_feasibility(self, vertex_key1, vertex_key2):
        """
        Check if a laser inter-satellite link (LISL) is feasible between two satellites.

        Parameters:
        - vertex_key1: Tuple (orbit_id, sat_id) of the first satellite.
        - vertex_key2: Tuple (orbit_id, sat_id) of the second satellite.

        Returns:
        - True if the LISL is feasible, False otherwise.
        """
        try:
            sat1 = self.graph.nodes[vertex_key1]["sat"]
            sat2 = self.graph.nodes[vertex_key2]["sat"]
        except KeyError:
            raise ValueError("One or both vertex keys do not exist in the network")
        lat1 = sat1.position_geodetic[0]
        lat2 = sat2.position_geodetic[0]

        # Check if the LISL is blocked by the Earth
        mid_point_ecef = (sat1.position_ecef + sat2.position_ecef) / 2
        earth_radius = EARTH_RADIUS
        alt_mid_point = (
            np.linalg.norm(mid_point_ecef) - earth_radius - 80
        )  # ! 80 km residue
        if alt_mid_point < 0:
            # The LISL is blocked by the Earth
            return False

        # * Check if the LISL subjects to the rules of the constellation
        if vertex_key1[0] == vertex_key2[0]:  # same orbit
            # Intra-plane LISL, assert that the satellites' ID are neighboring
            if (
                abs(
                    (vertex_key1[1] % self.constellation.num_sats_per_orbit)
                    - (vertex_key2[1] % self.constellation.num_sats_per_orbit)
                )
                == 1
            ):
                return True  # * actually, this should always hold true...
        else:
            #! this inter-orbit ISL logic might be modified
            # Inter-plane LISL, assert that:
            #   - the satellites are in adjacent orbits
            #   - the satellites are in the same latitude band, i.e., abs(lat1 - lat2) <= 1 #! don't understand this very much, there seems no such requirement at all
            #   - the satellites are not in the polar regions, i.e., abs(lat1 + lat2) / 2 <= gamma_deg
            if (
                abs(
                    (vertex_key1[0] % self.constellation.num_orbits)
                    - (vertex_key2[0] % self.constellation.num_orbits)
                )
                == 1
            ):
                if abs(lat1 - lat2) <= 1 and abs(lat1 + lat2) / 2 <= self.gamma_deg:
                    # Check if the latitudes are close and in range [-gamma_deg, gamma_deg]
                    return True

        return False


class SatNetDelta(SatNet):
    """
    Satellite network class for Walker Delta Constellations.
    """

    def __init__(self, constellation: DeltaConstellation):
        """
        Initialize the Walker Delta satellite network.

        Parameters:
        - constellation: DeltaConstellation object representing the satellite constellation.
        """
        if constellation.type != "Walker Delta Constellation":
            raise ValueError("SatNetDelta requires a Walker Delta Constellation")

        super().__init__(constellation)
        self._build_graph()
        return

    def update_graph(self):
        """Update the network graph for the Walker Delta Constellation."""
        self._build_graph()
        return

    def _build_graph(self):
        """
        Build the network graph for the Walker Delta Constellation.

        Returns:
        - graph: NetworkX graph representing the satellite network.
        """
        graph = nx.Graph()
        # Add nodes for each satellite in the constellation
        for orbit in self.constellation.orbits:
            for sat in orbit.sats:
                # Add node with key as (orbit_id, satellite_id) and store the satellite object
                graph.add_node((orbit.id, sat.id), sat=sat)
        self.graph = graph

        # Add edges for laser inter-satellite links
        # Intra-plane LISL: add edges between neighboring satellites in the same orbit
        for orbit in self.constellation.orbits:
            if orbit.num_sats < 2:
                continue  # No intra-plane links for single-satellite orbits
            for sat_id in range(1, orbit.num_sats + 1):
                sat1 = (orbit.id, sat_id)
                sat2 = (orbit.id, (sat_id % orbit.num_sats) + 1)
                if self._check_isl_feasibility(sat1, sat2):
                    distance = self.get_distance(sat1, sat2)
                    self.graph.add_edge(sat1, sat2, weight=distance)

        # Inter-plane LISL: add edges between neighboring satellites in adjacent orbits
        if self.constellation.num_orbits < 2:
            return self.graph  # No inter-plane links for single-orbit constellations
        for orbit_id in range(1, self.constellation.num_orbits + 1):
            if orbit_id != self.constellation.num_orbits:  # ! check point
                # For all orbits except the last one
                for sat_id in range(1, self.constellation.num_sats_per_orbit + 1):
                    sat1 = (orbit_id, sat_id)
                    sat2 = ((orbit_id % self.constellation.num_orbits) + 1, sat_id)
                    if self._check_isl_feasibility(sat1, sat2):
                        distance = self.get_distance(sat1, sat2)
                        self.graph.add_edge(sat1, sat2, weight=distance)
            else:
                # For the last orbit, add links to the first orbit, the offset is calculated based on the phasediff
                for sat_id in range(1, self.constellation.num_sats_per_orbit + 1):
                    sat_id_offset = (
                        self.constellation.num_orbits
                        * np.degrees(self.constellation.phasediff)
                    ) / (360 / self.constellation.num_sats_per_orbit)
                    # Round to the nearest integer
                    sat_id_offset = round(sat_id_offset)
                    sat1 = (
                        orbit_id,
                        (sat_id + sat_id_offset - 1)
                        % self.constellation.num_sats_per_orbit
                        + 1,
                    )
                    sat2 = (1, sat_id)
                    if self._check_isl_feasibility(sat1, sat2):
                        distance = self.get_distance(sat1, sat2)
                        self.graph.add_edge(sat1, sat2, weight=distance)

        return self.graph

    def _check_isl_feasibility(self, vertex_key1, vertex_key2):
        """
        Check if a laser inter-satellite link (LISL) is feasible between two satellites.

        Parameters:
        - vertex_key1: Tuple (orbit_id, sat_id) of the first satellite.
        - vertex_key2: Tuple (orbit_id, sat_id) of the second satellite.

        Returns:
        - True if the LISL is feasible, False otherwise.
        """
        try:
            sat1 = self.graph.nodes[vertex_key1]["sat"]
            sat2 = self.graph.nodes[vertex_key2]["sat"]
        except KeyError:
            raise ValueError("One or both vertex keys do not exist in the network")

        mid_point_ecef = (sat1.position_ecef + sat2.position_ecef) / 2
        earth_radius = EARTH_RADIUS
        alt_mid_point = np.linalg.norm(mid_point_ecef) - earth_radius
        if alt_mid_point < 0:
            # The LISL is blocked by the Earth
            return False

        return True


class ServiceSatNetDelta(SatNet):
    """
    Satellite network class for Walker Delta Constellations.
    """

    def __init__(
        self,
        constellation: ServiceDeltaConstellation,
        N_task,
        a_task,
        task_workload,
        link_bw,
        delta_t,
        max_time_span,
        bw_min,
        bw_max,
    ):
        """
        Initialize the Walker Delta satellite network.

        Parameters:
        - constellation: DeltaConstellation object representing the satellite constellation.
        """
        if constellation.type != "Walker Delta Constellation":
            raise ValueError("SatNetDelta requires a Walker Delta Constellation")
        super().__init__(constellation)
        self.original_constellation = copy.deepcopy(constellation)

        self.cities_init()

        self.link_bw = link_bw
        self.N = N_task
        # self.N_min = 1
        # self.N_max = 1
        self.N_min = int(np.ceil(N_task / 2))
        self.N_max = int(np.ceil(N_task * 1.5))
        self.a_min = a_task / 2
        self.a_max = 1.5 * a_task
        self.task_workload = task_workload
        self.delta_t = delta_t
        self.max_time_span = max_time_span
        self.train_time_unit_len = 1  # ! deprecated
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.M = 1e9
        self.epsilon = 1e-100

        # * random seed used for training and running
        self.train_seed = 64
        self.run_seed = 46

        # * auto-learning param
        self.new_task_alloc = []
        # self.T_unit = int(np.ceil(self.a_max / self.bw_min))
        # self.T_mem = 3 * self.T_unit  # 3
        # self.T_pred = 2 * self.T_unit  # 2

        self.max_propagation_delay = (
            2
            * np.pi
            * self.constellation.radius
            / (self.constellation.num_orbits)
            * self.constellation.num_sats
            / self.LIGHT_SPEED
        )
        self.T_mem = int(
            np.ceil(self.max_propagation_delay + self.a_max / self.bw_min)
        )  # 3
        self.T_max = self.T_mem
        self.T_pred = 2
        print(
            f"Delta: {self.delta_t}, T-memory: {self.T_mem}, T-predict: {self.T_pred}"
        )

        # self.T_train_data = self.train_time_unit_len * (self.T_mem + self.T_pred)
        # print(f"the time will be: {self.T_train_data}")

        self.V = 10
        self.m1 = 1 / self.constellation.num_isls if self.constellation.num_isls else 0
        self.m2 = 1 / self.constellation.num_sats
        self.m3 = 0.1 / (self.constellation.radius * self.a_max)
        self.m4 = 1

        self.nearest_record = result_record_class()
        self.nearest_gnd_record = result_record_class()

        self.lyap_MPC_record = result_record_class()
        self.vwq_matrix = np.zeros(
            [self.constellation.num_orbits, self.constellation.num_sats_per_orbit]
        )

        self.rand_record = result_record_class()
        self.rand_pwq_matrix = np.zeros(
            [self.constellation.num_orbits, self.constellation.num_sats_per_orbit]
        )

        self.get_constellation_haversine_matrix()
        self._init_graph()

        # target = Path("check_point")
        # for item in target.rglob("*"):
        #     if item.is_file():
        #         item.unlink()

    def cities_init(self):
        #! city coordinates: (lat, long, alt), using the Haversine distance to approximate
        cities = ["SFC", "Shanghai", "London"]
        self.sanfransisco_deg = (37.773972, -122.43129, self.constellation.radius)
        self.shanghai_deg = (31.22222, 121.45806, self.constellation.radius)
        self.london_deg = (51.509865, -0.12574, self.constellation.radius)
        self.cities_deg = [self.sanfransisco_deg, self.shanghai_deg, self.london_deg]
        self.cities_rad = np.deg2rad(self.cities_deg)
        self.cities_ecef = {}
        i = 0
        for city_rad in self.cities_rad:
            self.cities_ecef[cities[i]] = self.geodedic_to_ecef(city_rad)
            i += 1
        # print(f"cities of ecef here: {self.cities_ecef}")

    def geodedic_to_ecef(self, geo_coordinate_rad):
        lat, long, radius = (
            geo_coordinate_rad[0],
            geo_coordinate_rad[1],
            geo_coordinate_rad[2],
        )
        x = radius * np.cos(lat) * np.cos(long)
        y = radius * np.cos(lat) * np.sin(long)
        z = radius * np.sin(lat)
        return (x, y, z)

    def simu_task_arriving(self, task_num):
        """
        generate poisson arrivals for each task (with maximum truncate)
        """
        # * select random task arrivals
        selected_sat_indecies = self.rng.integers(
            self.constellation.num_sats, size=int(task_num)
        )  # random pick satellites
        task_size_vector = abs(
            self.rng.uniform(low=self.a_min, high=self.a_max, size=task_num)
        )  # generate service amount following normal distribution

        self.selected_sat_service = []
        for i in range(len(selected_sat_indecies)):
            # change the selected sat to the orbit id, sat id.
            s_orbit = selected_sat_indecies[i] // self.constellation.num_sats_per_orbit
            s_id = selected_sat_indecies[i] % self.constellation.num_sats_per_orbit

            self.selected_sat_service.append(
                (
                    (s_orbit, s_id, task_size_vector[i], self.task_workload),
                    task_size_vector[i] * self.task_workload,
                )
            )

    def cvx_create_TA_vars(self):
        self.cvx_task_alloc_var = {}
        for task in self.selected_sat_service:
            # * decision var, size = orbit * sat_per_orbit
            self.cvx_task_alloc_var[task[0]] = cp.Variable(
                (
                    self.constellation.num_orbits,
                    self.constellation.num_sats_per_orbit,
                ),
                # boolean=True,
            )

    def cvx_create_TA_obj(self):
        self.sat_vwq_vector = []
        self.sat_CPU_vector = []
        for orbit in self.constellation.orbits:
            self.sat_vwq_vector.append([sat.vwq for sat in orbit.sats])
            self.sat_CPU_vector.append([sat.fs for sat in orbit.sats])

        self.sat_vwq_vector = np.array(self.sat_vwq_vector)
        self.sat_CPU_vector = np.array(self.sat_CPU_vector)

        self.sum_of_alloc_service = cp.sum(
            [
                self.cvx_task_alloc_var[task[0]] * task[1]
                for task in self.selected_sat_service
                # * task[0]: (s_orbit, s_sat), task[1]: computational size = task size * task workload
            ]
        )

        self.haver_task = cp.sum(
            [  # ! haversine weighed offloading
                cp.multiply(
                    self.constellation_haversine_matrix[task[0][0]][task[0][1]],
                    self.cvx_task_alloc_var[task[0]],
                )
                for task in self.selected_sat_service
            ]
        )

        self.cvx_obj = cp.Minimize(
            self.V
            * self.m2
            * cp.sum_squares(
                cp.multiply(self.sum_of_alloc_service, 1 / self.sat_CPU_vector)
            )  # * quadratic load balancing
            + self.V * self.m3 * cp.sum(self.haver_task)
            + cp.sum(
                cp.multiply(self.sat_vwq_vector, self.sum_of_alloc_service)
            )  # * VWQ weighed load
        )

    def get_haversine_dist(self, sat1, sat2):
        # * using R=phi * r , with the ecef position to calculate haversine distance
        r1 = (
            np.asarray(sat1.position_ecef, dtype=float)
            if hasattr(sat1, "position_ecef")
            else sat1
        )
        r2 = (
            np.asarray(sat2.position_ecef, dtype=float)
            if hasattr(sat2, "position_ecef")
            else sat2
        )

        cos_theta = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        if hasattr(sat1, "radius"):
            radius = sat1.radius
        elif hasattr(sat2, "radius"):
            radius = sat2.radius
        else:
            radius = sat1[2]

        return radius * theta

    def get_haversine_matrix(self, source_sat):
        temp_matrix = np.zeros(
            [self.constellation.num_orbits, self.constellation.num_sats_per_orbit]
        )
        for orbit_id in range(len(self.constellation.orbits)):
            orbit = self.constellation.orbits[orbit_id]
            for sat_id in range(len(orbit.sats)):
                sat = orbit.sats[sat_id]
                temp_matrix[orbit_id][sat_id] = self.get_haversine_dist(source_sat, sat)
        return temp_matrix

    def get_constellation_haversine_matrix(self):
        self.constellation_haversine_matrix = []
        for orbit_id in range(len(self.constellation.orbits)):
            orbit = self.constellation.orbits[orbit_id]
            temp_orbit_mat = []
            for sat_id in range(len(orbit.sats)):
                sat = orbit.sats[sat_id]
                temp_orbit_mat.append(self.get_haversine_matrix(sat))
            self.constellation_haversine_matrix.append(temp_orbit_mat)

    def cvx_create_TA_constraints(self):
        self.cvx_constraints = (
            [
                0 <= v
                for task in self.selected_sat_service
                for v in self.cvx_task_alloc_var[task[0]]  # *
            ]
            + [
                v <= 1
                for task in self.selected_sat_service
                for v in self.cvx_task_alloc_var[task[0]]
            ]
            + [
                cp.sum(self.cvx_task_alloc_var[task[0]]) == 1
                for task in self.selected_sat_service
            ]
        )  # * task sum = 1

    def task_assignment_subproblem(self):
        # * create vars
        self.cvx_create_TA_vars()

        # * set obj
        self.cvx_create_TA_obj()

        # * set constr
        self.cvx_create_TA_constraints()

        # * construct the opt problem & solve
        self.cvx_full_prob = cp.Problem(self.cvx_obj, self.cvx_constraints)

        #! check point
        result = self.cvx_full_prob.solve(
            verbose=False,
            solver=cp.GUROBI,
            # IterationLimit=10
        )
        print(f"TA Result: {result}")
        return result
        # for task in self.selected_sat_service:
        #     print(f"decision var value of task{task[0]}, size: {task[1]}:\n{self.cvx_task_alloc_var[task[0]].value}") #* should be a matrix of all sats

    def vwq_update(self, check=0):
        """update the vwq w.r.t. decision var"""
        if len(self.selected_sat_service) > 0:
            temp_arrivals = [
                {
                    task: self.cvx_task_alloc_var[task[0]].value
                    * (self.cvx_task_alloc_var[task[0]].value > 1e-3)
                    * task[0][2]
                }
                for task in self.selected_sat_service
            ]

            workload_arrivals = cp.sum(
                [
                    self.cvx_task_alloc_var[task[0]].value
                    * (self.cvx_task_alloc_var[task[0]].value > 1e-3)
                    * task[1]
                    for task in self.selected_sat_service
                ]
            )

            if check:
                with open("check_point/vwq.txt", "a") as f:
                    f.write(
                        f"""====================\ntasks:\n{self.selected_sat_service}\nVWQ matrix:\n{self.vwq_matrix}\nworkload arrivals:\n{workload_arrivals}\n=====================\n"""
                    )  # ! check point

            for orbit_id in range(len(self.constellation.orbits)):
                orbit = self.constellation.orbits[orbit_id]
                for sat_id in range(len(orbit.sats)):
                    sat = orbit.sats[sat_id]
                    sat.vwq = (
                        max(0, sat.vwq - sat.fs) + workload_arrivals[orbit_id][sat_id]
                    )
                    self.vwq_matrix[orbit_id][sat_id] = sat.vwq
        else:
            temp_arrivals = []
            if check:
                with open("check_point/vwq.txt", "a") as f:
                    f.write(
                        f"""====================\ntasks:\n{self.selected_sat_service}\nVWQ matrix:\n{self.vwq_matrix}\nNO TASK HERE!!!\n=====================\n"""
                    )  # ! check point

            for orbit_id in range(len(self.constellation.orbits)):
                orbit = self.constellation.orbits[orbit_id]
                for sat_id in range(len(orbit.sats)):
                    sat = orbit.sats[sat_id]
                    sat.vwq = max(0, sat.vwq - sat.fs)
                    self.vwq_matrix[orbit_id][sat_id] = sat.vwq

        avg_vwq = 0
        for orbit_id in range(len(self.constellation.orbits)):
            orbit = self.constellation.orbits[orbit_id]
            for sat_id in range(len(orbit.sats)):
                sat = orbit.sats[sat_id]
                avg_vwq += (self.vwq_matrix[orbit_id][sat_id] / sat.fs) ** 2

        self.new_task_alloc.append(temp_arrivals)
        if len(self.new_task_alloc) > self.T_mem + 3:
            self.new_task_alloc.pop(0)

        return (
            avg_vwq / self.constellation.num_sats
        )  # ! this is only average VWQ, not the used PWQ

    def graph_reset(self):
        self.constellation = self.original_constellation
        self.vwq_matrix = np.zeros(
            [self.constellation.num_orbits, self.constellation.num_sats_per_orbit]
        )
        self._init_graph()

    def full_simu(self):
        """run the full time-span simulation"""
        #! remember to reinitialize the system fully from the training state
        self.graph_reset()

        self.rng = np.random.default_rng(
            seed=self.run_seed
        )  # ! train with seed = 64, predict with seed = 102

        # self.task_number_vector = self.rng.poisson(self.N, size=int(self.max_time_span))

        #! test task data
        # self.task_number_vector = np.int32(
        #     self.rng.uniform(
        #         low=self.N_min, high=self.N_max, size=int(self.max_time_span)
        #     )
        # )
        # self.task_number_vector = np.append(self.task_number_vector, 0)
        # self.task_number_vector = np.append(self.task_number_vector, 0)
        # self.task_number_vector = np.append(self.task_number_vector, 0)
        # self.task_number_vector = np.append(self.task_number_vector, 0)
        # self.task_number_vector = np.append(self.task_number_vector, 0)
        # self.task_number_vector = np.append(self.task_number_vector, 0)
        # self.task_number_vector = np.append(self.task_number_vector, 0)
        # print(f"task data: {self.task_number_vector}")

        # # * first of all
        # self.vectorized_bw_route_history = []
        # self.vectorized_trans_time_history = []
        # self.vectorized_workload_alloc_history = []
        # self.vectorized_BW_history = np.zeros(
        #     (self.constellation.num_sats, self.constellation.num_sats)
        # )
        # self.lyap_MPC_pwq_matrix = np.zeros((1, self.constellation.num_sats))
        # self.sat_processing_ability = np.array(
        #     [sat.fs for orbit in self.constellation.orbits for sat in orbit.sats]
        # )

        # for i in range(len(self.task_number_vector)):
        #     print(
        #         f":::::::::::::::::::THIS IS SLOT {i}/{self.max_time_span}::::::::::::::::::::"
        #     )
        #     self.simu_task_arriving(self.task_number_vector[i])
        #     lyap_MPC_cost = self.task_assignment_subproblem()
        #     self.vwq_update(check=0)
        #     self.vectorized_MPC(i)

    def V_test(self):
        # ! loop for const size, given task size & number
        with open("check_point/exp_data/lyap_MPC_record_V_test.json", "w") as f:
            pass

        self.original_V = self.V
        for self.V in range(0, 500, 50):
            self.new_task_alloc = []
            self.occupied_full_history = []

            self.nearest_gnd_full_history = []

            self.rand_full_history = []
            self.nearest_workload_matrix = np.zeros(
                [self.constellation.num_orbits, self.constellation.num_sats_per_orbit]
            )
            self.task_number_vector = np.int32(
                self.rng.uniform(
                    low=self.N_min, high=self.N_max, size=int(self.max_time_span)
                )
            )

            self.graph_reset()
            self.one_term_full_timescale_test(self.N, self.constellation.N_const, 1)

        # * save the data as local files
        #! check point
        with open("check_point/exp_data/lyap_MPC_record_V_test.json", "a") as f:
            json.dump(self.lyap_MPC_record.record_list, f)
        self.V = self.original_V

    def const_size_test(self):
        with open("check_point/exp_data/lyap_MPC_record_const_size.json", "w") as f:
            pass
        with open("check_point/exp_data/rand_record_const_size.json", "w") as f:
            pass
        with open("check_point/exp_data/nearest_record_const_size.json", "w") as f:
            pass
        with open("check_point/exp_data/nearest_gnd_record_const_size.json", "w") as f:
            pass

        # ! loop for const size, given task size & number
        self.N_max_original = self.N_max
        for N_const in range(5, 10, 1):
            self.new_task_alloc = []
            self.occupied_full_history = []

            self.nearest_gnd_full_history = []

            self.rand_full_history = []
            self.nearest_workload_matrix = np.zeros(
                [self.constellation.num_orbits, self.constellation.num_sats_per_orbit]
            )

            # self.task_number_vector = self.rng.poisson(
            #     N_const + 4, size=int(self.max_time_span)
            # )
            task_mean = N_const + 4
            self.N_max = int(np.ceil(task_mean * 1.5))

            self.task_number_vector = np.int32(
                self.rng.uniform(
                    low=task_mean * 0.5,
                    high=self.N_max,
                    size=int(self.max_time_span),
                )
            )
            self.constellation = ServiceDeltaConstellation(
                num_orbits=N_const,
                num_sats_per_orbit=N_const,
                radius=self.constellation.radius,
                inclination=self.constellation.inclination,
                phasediff=self.constellation.phasediff,
                fs=self.constellation.fs,
            )
            self.graph_reset()
            self.one_term_full_timescale_test(N_const + 4, N_const, 0)

        # * save the data as local files
        #! check point
        with open("check_point/exp_data/lyap_MPC_record_const_size.json", "a") as f:
            json.dump(self.lyap_MPC_record.record_list, f)
        with open("check_point/exp_data/rand_record_const_size.json", "a") as f:
            json.dump(self.rand_record.record_list, f)
        with open("check_point/exp_data/nearest_record_const_size.json", "a") as f:
            json.dump(self.nearest_record.record_list, f)
        with open("check_point/exp_data/nearest_gnd_record_const_size.json", "a") as f:
            json.dump(self.nearest_gnd_record.record_list, f)

    def task_num_test(self):
        with open("check_point/exp_data/lyap_MPC_record_task_num.json", "w") as f:
            pass
        with open("check_point/exp_data/rand_record_task_num.json", "w") as f:
            pass
        with open("check_point/exp_data/nearest_record_task_num.json", "w") as f:
            pass
        with open("check_point/exp_data/nearest_gnd_record_task_num.json", "w") as f:
            pass

        # ! loop for task number, given const size & task size
        self.N_max_original = self.N_max
        for task_num in range(8, 20, 1):
            self.N_max = int(np.ceil(task_num * 1.5))
            self.graph_reset()
            self.new_task_alloc = []
            self.occupied_full_history = []

            self.nearest_gnd_full_history = []

            self.rand_full_history = []
            self.nearest_workload_matrix = np.zeros(
                [self.constellation.num_orbits, self.constellation.num_sats_per_orbit]
            )

            # self.task_number_vector = self.rng.poisson(
            #     task_num, size=int(self.max_time_span)
            # )
            self.task_number_vector = np.int32(
                self.rng.uniform(
                    low=task_num / 2, high=self.N_max, size=int(self.max_time_span)
                )
            )

            self.one_term_full_timescale_test(task_num, self.constellation.N_const, 0)

        # * save the data as local files
        #! check point
        with open("check_point/exp_data/lyap_MPC_record_task_num.json", "a") as f:
            json.dump(self.lyap_MPC_record.record_list, f)
        with open("check_point/exp_data/rand_record_task_num.json", "a") as f:
            json.dump(self.rand_record.record_list, f)
        with open("check_point/exp_data/nearest_record_task_num.json", "a") as f:
            json.dump(self.nearest_record.record_list, f)
        with open("check_point/exp_data/nearest_gnd_record_task_num.json", "a") as f:
            json.dump(self.nearest_gnd_record.record_list, f)

    def vectorized_MPC(self, time_step):

        # * preliminaries
        self.prepare_graph_vectorized_parameters()
        self.prepare_task_vectorized_parameters()

        # * vectorized vars
        self.create_vectorized_MPC_cvx_vars()

        # * vectorized obj
        self.create_vectorized_MPC_cvx_obj()

        # * vectorized constr
        self.create_vectorized_MPC_cvx_constr(time_step)

        # * solve
        self.vectorized_MPC_cvx_problem = cp.Problem(
            self.vectorized_MPC_cvx_obj, self.vectorized_MPC_cvx_constr
        )
        MPC_cost = self.vectorized_MPC_cvx_problem.solve(
            verbose=False, solver=cp.GUROBI
        )
        print(f"VECTORIZED MPC ON!: {MPC_cost}")

        # * vectorized history update
        lyap_MPC_link_LB, lyap_MPC_sat_LB, lyap_MPC_MLLR, lyap_MPC_MSP = (
            self.vectorized_history_update()
        )

        return MPC_cost, lyap_MPC_link_LB, lyap_MPC_sat_LB, lyap_MPC_MLLR, lyap_MPC_MSP

    def vectorized_history_update(self):

        self.vectorized_BW_history = np.array(
            self.vectorized_BW[:, 0 : self.constellation.num_sats].value
        )

        final_decision = self.vectorized_auxiliary[
            :,
            0 : self.N_max * self.constellation.num_sats * self.constellation.num_sats,
        ].value
        final_route_decision = self.vectorized_routing_vector[
            :,
            0 : self.N_max * self.constellation.num_sats * self.constellation.num_sats,
        ].value

        if len(self.vectorized_bw_route_history) > 0:
            self.vectorized_bw_route_history = np.append(
                self.vectorized_bw_route_history, [final_decision], axis=0
            )
        else:
            self.vectorized_bw_route_history = np.array([final_decision])

        distance_mat = np.multiply(
            final_route_decision,
            np.kron(
                np.ones((1, self.constellation.num_sats * self.N_max)),
                self.vectorized_link_dist_mat,
            ),
        )
        full_distance_vector = []
        for task_id in range(self.N_max):
            task_start = task_id * self.constellation.num_sats
            for dst_sat_id in range(self.constellation.num_sats):
                dst_sat_start = task_start + dst_sat_id * self.constellation.num_sats
                full_distance_vector.append(
                    np.sum(
                        distance_mat[
                            :,
                            dst_sat_start : dst_sat_start + self.constellation.num_sats,
                        ]
                    )
                )
        full_distance_vector = np.array([full_distance_vector])
        vectorized_propagation_delay = full_distance_vector / self.LIGHT_SPEED

        # final_bw_decision = self.vectorized_bw_vector[
        #     0 : self.N_max * self.constellation.num_sats, :
        # ].value
        final_bw_decision = self.vectorized_bw_vector[
            0 : self.N_max * self.constellation.num_sats, :
        ]

        final_task_alloc = self.vectorized_task_alloc[
            :, 0 : self.N_max * self.constellation.num_sats
        ]
        vectorized_forwarding_delay = np.multiply(
            final_task_alloc, 1 / final_bw_decision.T
        )

        vectorized_full_task_trans_delay = np.ceil(
            vectorized_propagation_delay + vectorized_forwarding_delay
        )
        # * traverse all tasks, and set the time to 0 for self-offloading
        for tid, task in enumerate(self.future_task_list[0]):
            src_sat = task[0][0] * self.constellation.num_sats_per_orbit + task[0][1]
            vectorized_full_task_trans_delay[0][
                tid * self.constellation.num_sats + src_sat
            ] = 0
            final_bw_decision[tid * self.constellation.num_sats + src_sat][0] = 0

        if len(self.vectorized_trans_time_history) > 0:
            self.vectorized_trans_time_history = np.append(
                self.vectorized_trans_time_history,
                [vectorized_full_task_trans_delay],
                axis=0,
            )
        else:
            self.vectorized_trans_time_history = np.array(
                [vectorized_full_task_trans_delay]
            )

        # * MPC link LB
        link_load_ratio_mat = np.multiply(
            (
                self.vectorized_auxiliary[
                    :,
                    0 : self.N_max
                    * self.constellation.num_sats
                    * self.constellation.num_sats,
                ].value
                @ np.kron(
                    np.ones((self.N_max * self.constellation.num_sats, 1)),
                    np.eye(self.constellation.num_sats),
                )
            ),
            1 / (self.vectorized_full_link_bw_mat + self.epsilon),
        )
        lyap_MPC_link_LB = np.sum(link_load_ratio_mat**2) / len(self.graph.edges())
        lyap_MPC_MLLR = np.max(link_load_ratio_mat)

        final_workload_alloc = self.vectorized_workload_alloc
        self.vectorized_workload_alloc_history.append(final_workload_alloc)

        # if physical trans time == actual trans time, then update
        self.lyap_MPC_pwq_matrix = np.maximum(
            0, self.lyap_MPC_pwq_matrix - self.sat_processing_ability
        )
        for t_id in range(1, 1 + len(self.vectorized_trans_time_history)):
            update_indicator = self.vectorized_trans_time_history[-t_id] == (t_id - 1)
            # print(
            #     f"trans time history:\n{self.vectorized_trans_time_history[-t_id]}\ncheck update indicator:\n{update_indicator}\ntid now is: {t_id}\nworkload vector is\n{self.vectorized_workload_alloc_history[-t_id]}\nHello --- { np.multiply(update_indicator , self.vectorized_workload_alloc_history[-t_id])}"
            # )
            self.lyap_MPC_pwq_matrix += np.multiply(
                update_indicator, self.vectorized_workload_alloc_history[-t_id]
            ) @ np.kron(np.ones((self.N_max, 1)), np.eye(self.constellation.num_sats))

        lyap_MPC_MSP = np.max(self.lyap_MPC_pwq_matrix)
        lyap_MPC_sat_LB = np.sum(
            np.multiply(self.lyap_MPC_pwq_matrix, 1 / self.sat_processing_ability) ** 2
        ) / len(self.graph.nodes())
        print(
            f"!!!this is the pwq update checking: {self.lyap_MPC_pwq_matrix}, \nlyap_MPC_sat_LB: {lyap_MPC_sat_LB}, lyap_MPC_MSP: {lyap_MPC_MSP}"
        )

        if len(self.vectorized_bw_route_history) > 2 + self.T_mem:
            self.vectorized_bw_route_history = self.vectorized_bw_route_history[1:]
            self.vectorized_trans_time_history = self.vectorized_trans_time_history[1:]
            self.vectorized_workload_alloc_history = (
                self.vectorized_workload_alloc_history[1:]
            )

        # ! check point
        # print(
        #     f"trans time history:\n{vectorized_full_task_delay}\nRESULT CHECKING:\nfinal bw decision:\n {final_bw_decision}\nfinal route decision\n{final_route_decision}\nfinal bw-route:\n{final_decision}\nfinal task alloc:\n{final_task_alloc}\nvectorized forwarding delay:\n{vectorized_forwarding_delay}\nvectorized bw history:\n{self.vectorized_BW.value}"
        # )
        # print(
        #     f"RESULT CHECKING:\nfinal bw-route:\n{final_decision}\nvectorized bw history:\n{self.vectorized_BW.value}"
        # )
        return (lyap_MPC_link_LB, lyap_MPC_sat_LB, lyap_MPC_MLLR, lyap_MPC_MSP)

    def create_vectorized_MPC_cvx_constr(self, input_ts):
        # * bw min max constr
        self.vectorized_MPC_cvx_constr = []

        # * routing constr
        for time_step, task_list in enumerate(self.future_task_list):
            # print(f"time step is: {time_step}")
            start_time_id = time_step * self.N_max * self.constellation.num_sats
            for task_id, task_dt in enumerate(task_list):
                start_task_id = task_id * self.constellation.num_sats
                src_sat = (
                    task_dt[0][0] * self.constellation.num_sats_per_orbit
                    + task_dt[0][1]
                )
                # print(
                #     f"task id is:\n{task_id}, task_dt is: \n{task_dt[0][0], task_dt[0][1]}, src_sat is:\n{src_sat}"
                # )
                for i in range(self.constellation.num_sats):
                    # ! for each t, each task (src), each dst_sat, do the routing
                    dst_sat_id = start_time_id + start_task_id + i
                    # print(self.vectorized_task_alloc.shape)
                    # print(
                    #     f"checking: {self.vectorized_task_alloc[0][dst_sat_id], dst_sat_id}"
                    # )
                    if self.vectorized_task_alloc[0][dst_sat_id] and (
                        i != src_sat
                    ):  # ! if needs offloading
                        #! idea, collect all data and concatenate them?
                        # routing constr
                        #     print(
                        #         f"{dst_sat_id} is offloading: {self.vectorized_task_alloc[0][
                        #     dst_sat_id
                        # ]}"
                        # )
                        row_sum = cp.sum(
                            cp.multiply(
                                self.vectorized_conn_mat,
                                self.vectorized_routing_vector[
                                    :,
                                    dst_sat_id
                                    * self.constellation.num_sats : (dst_sat_id + 1)
                                    * self.constellation.num_sats,
                                ],
                            ),
                            axis=0,
                        )  # in flow
                        column_sum = cp.sum(
                            cp.multiply(
                                self.vectorized_conn_mat,
                                self.vectorized_routing_vector[
                                    :,
                                    dst_sat_id
                                    * self.constellation.num_sats : (dst_sat_id + 1)
                                    * self.constellation.num_sats,
                                ],
                            ),
                            axis=1,
                        )  # out flow
                        offset_vector = np.zeros((1, self.constellation.num_sats))
                        offset_vector[0][src_sat] = 1  # * src offset
                        offset_vector[0][i] = -1  # * dst offset
                        self.vectorized_MPC_cvx_constr += [
                            column_sum == row_sum.T + offset_vector.flatten().T,
                            column_sum[src_sat] == 1,
                            row_sum[i] == 1,
                        ]
                        # print(f"offset vector:\n{offset_vector}, row_sum vector:\n{row_sum}")
                    else:  # not offload
                        self.vectorized_MPC_cvx_constr += [
                            self.vectorized_routing_vector[
                                :,
                                dst_sat_id
                                * self.constellation.num_sats : (dst_sat_id + 1)
                                * self.constellation.num_sats,
                            ]
                            == 0
                        ]

        # * mccormick envelop constr
        base_one_mat = np.ones(
            (self.constellation.num_sats, self.constellation.num_sats)
        )
        large_base_one_mat = np.ones(
            (  # * same size as routing vector
                self.constellation.num_sats,
                self.T_pred
                * self.N_max
                * self.constellation.num_sats
                * self.constellation.num_sats,
            )
        )

        # * bw evolution
        self.vectorized_BW = []
        self.vectorized_trans_time_auxiliary = []
        self.vectorized_second_auxiliary = []
        for tp in range(
            self.T_pred
        ):  # ! for each prediction time step, a set of vectorized constraints and metrics are given
            leaving_bw_vector = []
            # print(f"PRIDICTING TIME: {tp}!!!!")
            for tm in range(
                1, self.T_mem + 1
            ):  # ! calculate the corresponding leaving bw
                mem_time_step = tp + input_ts - tm
                if mem_time_step >= 0:
                    if mem_time_step < input_ts:  # ! has memory
                        leaving_indicator_vector = (
                            self.vectorized_trans_time_history[tp - tm] == tm
                        )
                        # print(f"PRIDICTING TIME: {tp}!!!!, the leaving indicator from history: {leaving_indicator_vector}, tm-tp: {tm-tp}, and {self.vectorized_trans_time_history}")
                        # print(f"leaving bw vector: \n{leaving_indicator_vector}")
                        leaving_selection_vector = np.kron(
                            leaving_indicator_vector,
                            np.eye(self.constellation.num_sats),
                        )
                        leaving_bw_vector.append(
                            self.vectorized_bw_route_history[tp - tm]
                            @ leaving_selection_vector.T
                        )

                    else:  #! future
                        # ! calculate future propagation time
                        # print(f"FUTURE IS HERE!!!!!!!!!!!!!!")
                        future_routing = self.vectorized_routing_vector[
                            :,
                            (tp - tm)
                            * self.N_max
                            * self.constellation.num_sats
                            * self.constellation.num_sats : (tp - tm + 1)
                            * self.N_max
                            * self.constellation.num_sats
                            * self.constellation.num_sats,
                        ]
                        future_full_distance = cp.multiply(
                            future_routing,
                            np.kron(
                                np.ones((1, self.N_max * self.constellation.num_sats)),
                                self.vectorized_link_dist_mat,
                            ),
                        )

                        future_distance_vector = (
                            np.ones((1, self.constellation.num_sats))
                            @ future_full_distance
                            @ np.kron(
                                np.eye(self.N_max * self.constellation.num_sats),
                                np.ones((self.constellation.num_sats, 1)),
                            )
                        )
                        future_propagation_delay = (
                            future_distance_vector / self.LIGHT_SPEED
                        )

                        # ! calculate future forwarding time: data / bw
                        future_forwarding_delay = np.multiply(
                            self.vectorized_task_alloc[
                                0,
                                (tp - tm)
                                * self.N_max
                                * self.constellation.num_sats : (tp - tm + 1)
                                * self.N_max
                                * self.constellation.num_sats,
                            ],
                            # 1
                            # / self.vectorized_bw_vector[
                            #     (tp - tm)
                            #     * self.N_max
                            #     * self.constellation.num_sats : (tp - tm + 1)
                            #     * self.N_max
                            #     * self.constellation.num_sats
                            # ].T,
                            1
                            / self.bw_min  # ! warning: another approximation is used: bw_min to replace the var bw
                            * np.ones((1, self.N_max * self.constellation.num_sats)),
                        )

                        future_trans_time = (
                            future_propagation_delay + future_forwarding_delay
                        )

                        # ! big-M reformulation: for release time indicator, because the propagation delay depends on the routing decision (what if propagation delay is relaxed ot the max... not possible)
                        # leaving_indicator_vector = cp.kron(
                        #     future_trans_time == tp - tm,
                        #     np.eye(self.constellation.num_sats),
                        # )
                        offload_indicator = (
                            self.vectorized_task_alloc[
                                0,
                                (tp - tm)
                                * self.N_max
                                * self.constellation.num_sats : (tp - tm + 1)
                                * self.N_max
                                * self.constellation.num_sats,
                            ]
                            > 1e-5
                        ).astype(int)

                        self.vectorized_trans_time_auxiliary.append(
                            cp.Variable(
                                (self.T_max, self.N_max * self.constellation.num_sats),
                                boolean=True,
                            )
                        )
                        # * no future MPC
                        self.vectorized_MPC_cvx_constr += [
                            (
                                np.ones((1, self.T_max))
                                @ self.vectorized_trans_time_auxiliary[-1]
                            )
                            == offload_indicator,
                        ]
                        for i in range(self.T_pred):
                            self.vectorized_MPC_cvx_constr += [
                                future_trans_time
                                <= i
                                + self.M
                                * (1 - self.vectorized_trans_time_auxiliary[-1][i]),
                                future_trans_time
                                >= i
                                - 1
                                + self.epsilon
                                - self.M
                                * (1 - self.vectorized_trans_time_auxiliary[-1][i]),
                            ]

                        # ! this part leads to a second-level bilinear term: because the release term depends on both bw alloc and release time, which needs a second McCormick envelopes
                        # leaving_bw_vector.append(
                        #     self.vectorized_auxiliary[
                        #         :,
                        #         (tp - tm)
                        #         * self.N_max
                        #         * self.constellation.num_sats
                        #         * self.constellation.num_sats : (tp - tm + 1)
                        #         * self.N_max
                        #         * self.constellation.num_sats
                        #         * self.constellation.num_sats,
                        #     ]
                        #     @ cp.kron(
                        #         cp.reshape(
                        #             self.vectorized_trans_time_auxiliary[-1][tm],
                        #             (1, self.N_max * self.constellation.num_sats),
                        #             order="C",
                        #         ),
                        #         base_one_mat, and this should not be the base_one_mat, however should be the identity matrix of the same size
                        #     ).T
                        # )
                        # ! 1. do the element-wise mccormick envelop
                        self.vectorized_second_auxiliary.append(
                            cp.Variable(
                                (
                                    self.constellation.num_sats,
                                    self.N_max
                                    * self.constellation.num_sats
                                    * self.constellation.num_sats,
                                )
                            )
                        )
                        self.vectorized_MPC_cvx_constr += (
                            [self.vectorized_second_auxiliary[-1] >= 0]
                            + [
                                self.vectorized_second_auxiliary[-1]
                                >= self.bw_min
                                * cp.kron(
                                    cp.reshape(
                                        self.vectorized_trans_time_auxiliary[-1][tm],
                                        (1, self.N_max * self.constellation.num_sats),
                                        order="C",
                                    ),
                                    base_one_mat,
                                )
                                + self.vectorized_auxiliary[
                                    :,
                                    (tp - tm)
                                    * self.N_max
                                    * self.constellation.num_sats
                                    * self.constellation.num_sats : (tp - tm + 1)
                                    * self.N_max
                                    * self.constellation.num_sats
                                    * self.constellation.num_sats,
                                ]
                                - self.bw_min
                                * np.ones(
                                    (
                                        self.constellation.num_sats,
                                        self.N_max
                                        * self.constellation.num_sats
                                        * self.constellation.num_sats,
                                    )
                                )
                            ]
                            + [
                                self.vectorized_second_auxiliary[-1]
                                <= self.bw_min
                                * cp.kron(
                                    cp.reshape(
                                        self.vectorized_trans_time_auxiliary[-1][tm],
                                        (1, self.N_max * self.constellation.num_sats),
                                        order="C",
                                    ),
                                    base_one_mat,
                                )
                            ]
                            + [
                                self.vectorized_second_auxiliary[-1]
                                <= self.vectorized_auxiliary[
                                    :,
                                    (tp - tm)
                                    * self.N_max
                                    * self.constellation.num_sats
                                    * self.constellation.num_sats : (tp - tm + 1)
                                    * self.N_max
                                    * self.constellation.num_sats
                                    * self.constellation.num_sats,
                                ]
                            ]
                        )
                        # ! 2. do the block-sum
                        leaving_bw_vector.append(
                            self.vectorized_second_auxiliary[-1]
                            @ np.kron(
                                np.ones((1, self.N_max * self.constellation.num_sats)),
                                np.eye(self.constellation.num_sats),
                            ).T
                        )

            # BW[t+1] = BW[t] - leaving_bw_vector + arriving_vector <= BW^max
            if len(leaving_bw_vector) == 0:
                leaving_bw_vector = np.array([0])
                full_leaving_bw_vector = 0
            elif len(leaving_bw_vector) == 1:
                full_leaving_bw_vector = leaving_bw_vector[0]
            else:
                full_leaving_bw_vector = leaving_bw_vector[0]
                for item in leaving_bw_vector[1:]:
                    full_leaving_bw_vector += item
                # print(
                #     f"shape shape shape!!!!!!!!!!!!!!!_______{full_leaving_bw_vector.shape}_______"
                # )

            # print(f"xxxxxxxxxxxx full leaving bw: {full_leaving_bw_vector} xxxxxxxxxxxxxx")
            if tp == 0:  # ! bw of the first slot is the last time
                self.vectorized_BW.append(copy.deepcopy(self.vectorized_BW_history))
                # print(f"LAST TIME:\n{self.vectorized_BW}")
            else:
                self.vectorized_BW.append(self.vectorized_BW[-1])

            # print(
            #     f"SHAPE CHECK: bw - {self.vectorized_BW[-1].shape}, vectorized_bw_history - {self.vectorized_BW_history.shape}"
            # )

            self.vectorized_BW[-1] = self.vectorized_BW[-1] + (
                self.vectorized_auxiliary[
                    :,
                    tp
                    * self.N_max
                    * self.constellation.num_sats
                    * self.constellation.num_sats : (tp + 1)
                    * self.N_max
                    * self.constellation.num_sats
                    * self.constellation.num_sats,
                ]
                @ np.kron(
                    np.ones((1, self.N_max * self.constellation.num_sats)),
                    np.eye(self.constellation.num_sats),
                ).T
                - full_leaving_bw_vector
            )

            # self.vectorized_MPC_cvx_constr += [
            #     self.vectorized_BW[-1] - full_leaving_bw_vector <= self.vectorized_full_link_bw_mat
            #     # self.vectorized_BW[-1] <= self.vectorized_full_link_bw_mat
            # ]

        self.vectorized_BW = cp.hstack(self.vectorized_BW)
        # ! test comment
        self.vectorized_MPC_cvx_constr += [
            self.vectorized_BW
            <= np.kron(np.ones((1, self.T_pred)), self.vectorized_full_link_bw_mat)
        ]

    def create_vectorized_MPC_cvx_obj(self):
        load_balancing_val = cp.sum_squares(
            cp.multiply(
                self.vectorized_routing_vector
                @ cp.kron(
                    self.vectorized_bw_vector,
                    np.eye(self.constellation.num_sats),
                ),
                1 / (self.vectorized_full_link_bw_mat + self.epsilon),
            )
        )
        route_len = cp.sum(self.vectorized_routing_vector)

        self.vectorized_MPC_cvx_obj = cp.Minimize(
            self.m1 * load_balancing_val + self.m4 * route_len
        )

    def find_sat_index(self, sat):
        return (sat[0] - 1) * self.constellation.num_sats_per_orbit + (sat[1] - 1)

    def prepare_task_vectorized_parameters(self):
        # vectorize task alloc result
        self.vectorized_task_alloc = np.zeros(
            (1, self.T_pred * self.N_max * self.constellation.num_sats)
        )
        self.vectorized_workload_alloc = np.zeros(
            (1, self.N_max * self.constellation.num_sats)
        )

        self.future_task_list = []

        future_task = []
        for idx, dt in enumerate(self.new_task_alloc[-1]):
            for key, val in dt.items():
                temp_offload = val.flatten()
                future_task.append(key)
                for i, d in enumerate(temp_offload):
                    self.vectorized_task_alloc[0][
                        idx * self.constellation.num_sats + i
                    ] = d
                    self.vectorized_workload_alloc[0][
                        idx * self.constellation.num_sats + i
                    ] = (d * key[0][3])
        # * record the task src data
        self.future_task_list.append(future_task)

        # duplicate the past alloc as future
        future_padding = np.zeros((1, self.N_max * self.constellation.num_sats))
        future_task = []
        for x in reversed(self.new_task_alloc):
            if x:
                for idx, dt in enumerate(x):
                    for key, val in dt.items():
                        temp_offload = val.flatten()
                        future_task.append(key)
                        for i, d in enumerate(temp_offload):
                            future_padding[0][idx * self.constellation.num_sats + i] = d
                break

        future_indicator = np.array([np.r_[0, np.ones(self.T_pred - 1)]])
        self.vectorized_task_alloc += np.kron(future_indicator, future_padding)
        for i in range(self.T_pred - 1):
            self.future_task_list.append(future_task)
        # print(f"future task ths time & : {self.vectorized_task_alloc}")

    def prepare_graph_vectorized_parameters(self):
        self.vectorized_full_link_bw_mat = np.zeros(
            (self.constellation.num_sats, self.constellation.num_sats)
        )
        self.vectorized_link_dist_mat = np.zeros(
            (self.constellation.num_sats, self.constellation.num_sats)
        )
        self.vectorized_conn_mat = np.zeros(
            (self.constellation.num_sats, self.constellation.num_sats)
        )
        for s, d in self.graph.edges():
            s_sat = self.find_sat_index(s)
            d_sat = self.find_sat_index(d)
            self.vectorized_full_link_bw_mat[s_sat][d_sat] = self.graph[s][d]["full_bw"]
            self.vectorized_link_dist_mat[s_sat][d_sat] = self.graph[s][d]["distance"]
            self.vectorized_conn_mat[s_sat][d_sat] = 1

        self.vectorized_bw_record = copy.deepcopy(self.vectorized_full_link_bw_mat)
        # print(f"link conn mat:\n{self.vectorized_conn_mat}")

    def create_vectorized_MPC_cvx_vars(self):
        self.vectorized_bw_vector = self.bw_min * np.ones(
            (self.T_pred * self.N_max * self.constellation.num_sats, 1)
        )

        self.vectorized_routing_vector = cp.Variable(
            (
                self.constellation.num_sats,
                self.T_pred
                * self.N_max
                * self.constellation.num_sats
                * self.constellation.num_sats,
            ),
            boolean=True,
        )

        self.vectorized_auxiliary = cp.multiply(
            self.vectorized_routing_vector,
            np.kron(
                self.vectorized_bw_vector.T,
                np.ones((self.constellation.num_sats, self.constellation.num_sats)),
            ),
        )

    def one_term_full_timescale_test(self, task_num, N_const, V_test):
        # * do the full time-scale simu on the given param

        # * first of all
        self.vectorized_bw_route_history = []
        self.vectorized_trans_time_history = []
        self.vectorized_workload_alloc_history = []
        self.vectorized_BW_history = np.zeros(
            (self.constellation.num_sats, self.constellation.num_sats)
        )
        self.lyap_MPC_pwq_matrix = np.zeros((1, self.constellation.num_sats))
        self.sat_processing_ability = np.array(
            [sat.fs for orbit in self.constellation.orbits for sat in orbit.sats]
        )

        for i in range(len(self.task_number_vector)):
            # for i in range(8):
            print(
                f":::::::::::::::::::THIS IS ({"IS" if V_test else "NOT"} V-test, V is {self.V}): {task_num} tasks, {N_const}-Constellation, SLOT {i}/{self.max_time_span}::::::::::::::::::::"
            )
            start_time = time.perf_counter()

            # # * new service request generation to each sat
            self.simu_task_arriving(self.task_number_vector[i])

            # ! lyap-MPC simu ===============================================
            print(f"------------------This is Lyap-MPC-------------------")
            # * solve the task assignment problem
            lyap_cost = self.task_assignment_subproblem()

            # * update vwq
            self.vwq_update(check=0)

            # * solve the MPC communication problem
            # lyap_MPC_cost += self.solve_MPC_comm(i)
            MPC_cost, lyap_MPC_link_LB, lyap_MPC_sat_LB, lyap_MPC_MLLR, lyap_MPC_MSP = (
                self.vectorized_MPC(i)
            )

            # * data collection
            self.lyap_MPC_record.add_record(
                lyap_cost + MPC_cost,
                lyap_MPC_link_LB,
                lyap_MPC_sat_LB,
                lyap_MPC_MLLR,
                lyap_MPC_MSP,
                self.a_min,
                self.a_max,
                task_num,
                self.V,
                N_const,
            )

            if not V_test:
                # ! rand simu ===============================================
                print(f"------------------This is RAND-------------------")
                self.do_rand_task_alloc()  # * rand alloc task
                self.do_rand_bw_alloc()  # * rand alloc bw
                (rand_link_LB, rand_MLLR, rand_sat_LB, rand_MSP) = (
                    self.rand_bw_update()
                )  # * rand bw update

                # * rand data collection
                self.rand_record.add_record(
                    None,
                    rand_link_LB,
                    rand_sat_LB,
                    rand_MLLR,
                    rand_MSP,
                    self.a_min,
                    self.a_max,
                    task_num,
                    None,
                    N_const,
                )

                # * nearest-only simu ====================================
                # nearest-only only offloads to the coming sat itself
                # so, only workload exists
                print(f"------------------This is Nearest sat-------------------")
                nearest_sat_LB, nearest_MSP = self.nearest_workload_update()
                self.nearest_record.add_record(
                    None,
                    None,
                    nearest_sat_LB,
                    None,
                    nearest_MSP,
                    self.a_min,
                    self.a_max,
                    task_num,
                    None,
                    N_const,
                )

                # * ground-only simu
                # only offloads for ground (cloud) processing
                # thus, only transmission is needed
                print(f"------------------This is Nearest Ground-------------------")
                nearest_gnd_link_LB, nearest_gnd_MLLR = self.nearest_gnd_update()
                self.nearest_gnd_record.add_record(
                    None,
                    nearest_gnd_link_LB,
                    None,
                    nearest_gnd_MLLR,
                    None,
                    self.a_min,
                    self.a_max,
                    task_num,
                    None,
                    N_const,
                )

            self.update_graph()  # * update the nwk graph
            elapsed = time.perf_counter() - start_time
            print(
                f"=============ELAPSED TIME FOR THIS ROUND: {elapsed: .4f}s================"
            )

    def nearest_gnd_update(self):
        # * 1. find the nearest access sat for each city
        nearest_access_sat = {}
        for key, val in self.cities_ecef.items():
            nearest_access_sat[key] = {
                "dist": 1e20,
                "sat": None,
            }
            for o_id in range(len(self.constellation.orbits)):
                orbit = self.constellation.orbits[o_id]
                for s_id in range(len(orbit.sats)):
                    sat = orbit.sats[s_id]
                    dist = self.get_haversine_dist(sat, val)
                    if dist < nearest_access_sat[key]["dist"]:
                        nearest_access_sat[key]["dist"] = dist
                        nearest_access_sat[key]["sat"] = (o_id + 1, s_id + 1)
        # print(f"cities: {nearest_access_sat}")

        self.nearest_gnd_full_history.append([])
        for task in self.selected_sat_service:
            src_sat = (int(task[0][0] + 1), int(task[0][1] + 1))
            if_offload = 1
            offload_sat_dist = 1e30
            offload_sat = None
            for key, val in nearest_access_sat.items():
                if val["sat"] == src_sat:
                    if_offload = 0
                    offload_sat_dist = 0
                else:  # * find the offload access sat
                    dist = self.get_haversine_dist(
                        self.graph.nodes[src_sat]["sat"],
                        self.graph.nodes[val["sat"]]["sat"],
                    )
                    if offload_sat_dist > dist:
                        offload_sat = val["sat"]
                        offload_sat_dist = dist

            # print(f"offload to {offload_sat}")

            if if_offload:
                residue_bw = {}
                # * for each taslk, a new graph is needed
                self.nearest_gnd_temp_graph = nx.Graph()
                for s, d in self.graph.edges():
                    residue_bw[(s, d)] = (
                        self.graph[s][d]["full_bw"] - self.nearest_gnd_bw_vector[(s, d)]
                    )
                    if residue_bw[(s, d)] >= self.bw_min:
                        dist = self.graph[s][d]["distance"]
                        self.nearest_gnd_temp_graph.add_edge(s, d, distance=dist)

                path = nx.shortest_path(
                    self.nearest_gnd_temp_graph, src_sat, offload_sat, weight="distance"
                )
                path_len = nx.shortest_path_length(
                    self.nearest_gnd_temp_graph, src_sat, offload_sat, weight="distance"
                )
                propagation_delay = path_len / self.LIGHT_SPEED

                temp_bw = self.bw_min
                trans_delay = task[0][2] / temp_bw

                for i in range(len(path) - 1):
                    self.nearest_gnd_bw_vector[(path[i], path[i + 1])] += temp_bw

                self.nearest_gnd_full_history[-1].append(
                    {
                        "task_description": task[0],
                        "trans_time": np.ceil(trans_delay + propagation_delay),
                        "route": {},
                        "bw": temp_bw,
                    }
                )

                for s, d in self.graph.edges():
                    self.nearest_gnd_full_history[-1][-1]["route"][(s, d)] = 0
                for i in range(len(path) - 1):
                    self.nearest_gnd_full_history[-1][-1]["route"][
                        (path[i], path[i + 1])
                    ] = 1

        # bw update
        for t in range(2, 2 + self.T_mem):
            if t <= len(self.nearest_gnd_full_history):
                for task_id in range(len(self.nearest_gnd_full_history[-t])):
                    task = self.nearest_gnd_full_history[-t][task_id]
                    if task["trans_time"] == (t - 1):
                        for key, val in task["route"].items():
                            if val:
                                self.nearest_gnd_bw_vector[key] -= task["bw"]

        if len(self.nearest_gnd_full_history) > (2 + self.T_mem):
            self.nearest_gnd_full_history.pop(0)

        # metrics calculation
        nearest_gnd_link_LB = 0
        nearest_gnd_MLLR = 0
        for s, d in self.graph.edges():
            link_ratio = (
                self.nearest_gnd_bw_vector[(s, d)] / self.graph[s][d]["full_bw"]
            )
            nearest_gnd_link_LB += (link_ratio) ** 2
            nearest_gnd_MLLR = max(nearest_gnd_MLLR, link_ratio)

        # print(f"bw occupation: {self.nearest_gnd_bw_vector}")
        return nearest_gnd_link_LB, nearest_gnd_MLLR

    def nearest_workload_update(self):
        for o_id in range(len(self.nearest_workload_matrix)):
            for s_id in range(len(self.nearest_workload_matrix[o_id])):
                s_fs = self.graph.nodes[(o_id + 1, s_id + 1)]["fs"]
                self.nearest_workload_matrix[o_id][s_id] = max(
                    0, self.nearest_workload_matrix[o_id][s_id] - s_fs
                )
        for task in self.selected_sat_service:
            orbit_id = task[0][0]
            sat_id = task[0][1]
            self.nearest_workload_matrix[orbit_id][sat_id] += task[1]

        nearest_sat_LB = 0
        nearest_MSP = 0
        for o_id in range(len(self.nearest_workload_matrix)):
            for s_id in range(len(self.nearest_workload_matrix[o_id])):
                s_fs = self.graph.nodes[(o_id + 1, s_id + 1)]["fs"]
                nearest_sat_LB += (self.nearest_workload_matrix[o_id][s_id] / s_fs) ** 2
                nearest_MSP += max(
                    nearest_sat_LB, self.nearest_workload_matrix[o_id][s_id]
                )
        return nearest_sat_LB, nearest_MSP

    def rand_bw_update(self):
        rand_link_LB = 0
        rand_sat_LB = 0
        rand_MLLR = 0
        rand_MSP = 0
        # new_arrivals = {}
        history_leavings = {}
        # print(f"state checking: {self.rand_occupied_bw_vector}")

        for s, d in self.graph.edges():
            history_leavings[(s, d)] = 0
            for dst_sat in self.graph.nodes():
                # * calculate history leavings
                for t in range(2, 2 + self.T_mem):
                    if t > len(self.rand_full_history):
                        break
                    for task_id in range(len(self.rand_full_history[-t])):
                        task_data = self.rand_full_history[-t][task_id][dst_sat]
                        if task_data["trans_time"] == (t - 1):
                            history_leavings[(s, d)] += (
                                task_data["route"][(s, d)] * task_data["bw"]
                            )
                            #! check point
                            # print(f"task bw alloc: {task_data["bw"]}, task data amount: {task_data["offload_amount"]}")

            self.rand_occupied_bw_vector[(s, d)] -= history_leavings[(s, d)]
            if self.rand_occupied_bw_vector[(s, d)] < 1e-4:
                self.rand_occupied_bw_vector[(s, d)] = 0
            # self.rand_occupied_bw_vector[(s, d)] += (
            #     new_arrivals[(s, d)] - history_leavings[(s, d)]
            # )
            link_ratio = (
                self.rand_occupied_bw_vector[(s, d)] / self.graph[s][d]["full_bw"]
            )
            rand_link_LB += (link_ratio) ** 2
            rand_MLLR = max(rand_MLLR, link_ratio)
        rand_link_LB /= len(self.graph.edges())
        # print(f"leaving checking: {history_leavings}")

        # * update PWQ
        for dst_sat in self.graph.nodes():
            dst_orbit = dst_sat[0] - 1
            dst_sid = dst_sat[1] - 1
            self.rand_pwq_matrix[dst_orbit][dst_sid] = max(
                0,
                self.rand_pwq_matrix[dst_orbit][dst_sid]
                - self.graph.nodes[dst_sat]["fs"],
            )
            for t in range(2, 2 + self.T_mem):
                if t > len(self.rand_full_history):
                    break
                for task_id in range(len(self.rand_full_history[-t])):
                    task_data = self.rand_full_history[-t][task_id][dst_sat]
                    if task_data["trans_time"] == (t - 1):
                        self.rand_pwq_matrix[dst_orbit][dst_sid] += task_data[
                            "offload_amount"
                        ]

            rand_sat_LB += (
                self.rand_pwq_matrix[dst_orbit][dst_sid]
                / self.graph.nodes[dst_sat]["fs"]
            ) ** 2
            rand_MSP = max(rand_MSP, self.rand_pwq_matrix[dst_orbit][dst_sid])
        rand_sat_LB /= len(self.graph.nodes())

        if len(self.rand_full_history) > 2 + self.T_mem:
            self.rand_full_history.pop(0)

        return rand_link_LB, rand_MLLR, rand_sat_LB, rand_MSP

    def do_rand_bw_alloc(self):
        self.rand_full_history.append([])
        #! for each service, each sat, record: size, bw, trans_time
        for key, val in self.rand_task_alloc.items():
            task_description = key
            task_data = val * task_description[2]
            src_sat = (task_description[0] + 1, task_description[1] + 1)
            self.rand_full_history[-1].append({})  # * each task, a dict

            for sat in self.graph.nodes():
                offload_amount = task_data[sat[0] - 1][sat[1] - 1]
                if (sat != src_sat) and (
                    offload_amount > 1e-3
                ):  # * if needed offloading
                    #! choose bw and route accomodate to the bw residue
                    self.rand_temp_graph = nx.DiGraph()
                    rand_bw_residue = {}
                    for s, d in self.graph.edges():
                        rand_bw_residue[(s, d)] = (
                            self.graph[s][d]["full_bw"]
                            - self.rand_occupied_bw_vector[(s, d)]
                        )
                        if rand_bw_residue[(s, d)] >= self.bw_min:
                            self.rand_temp_graph.add_edge(
                                s,
                                d,
                                distance=self.graph[s][d]["distance"],
                            )

                    # randomly choose an available bw val
                    path = nx.shortest_path(
                        self.rand_temp_graph, src_sat, sat, weight="distance"
                    )

                    temp_bw = self.bw_min

                    path_len = nx.shortest_path_length(
                        self.graph, src_sat, sat, weight="distance"
                    )
                    propagation_delay = path_len / self.LIGHT_SPEED
                    trans_delay = offload_amount / temp_bw
                    self.rand_full_history[-1][-1][sat] = {
                        "offload_amount": offload_amount,
                        "bw": temp_bw,
                        "trans_time": np.ceil(trans_delay + propagation_delay),
                        "route": {},
                    }

                    for s, d in self.graph.edges():
                        self.rand_full_history[-1][-1][sat]["route"][(s, d)] = 0

                    for i in range(len(path) - 1):
                        self.rand_full_history[-1][-1][sat]["route"][
                            (path[i], path[i + 1])
                        ] = 1
                        self.rand_occupied_bw_vector[(path[i], path[i + 1])] += temp_bw

                else:
                    self.rand_full_history[-1][-1][sat] = {
                        "offload_amount": 0,
                        "bw": 0,
                        "trans_time": 0,
                        "route": {},
                    }
                    for s, d in self.graph.edges():
                        self.rand_full_history[-1][-1][sat]["route"][(s, d)] = 0

    def do_rand_task_alloc(self):
        self.rand_task_alloc = {}
        for task in self.selected_sat_service:
            rand_ratio = self.rng.random(
                self.constellation.num_orbits * self.constellation.num_sats_per_orbit
                - 1
            )
            self.rand_task_alloc[task[0]] = np.append(
                rand_ratio, 1 - np.sum(rand_ratio)
            ).reshape(
                (
                    self.constellation.num_orbits,
                    self.constellation.num_sats_per_orbit,
                )
            )

    # * get orbit & sat info
    def const_info_checking(self, check: int):
        all_sat_position = {}
        all_orbit_info = {}

        for orbit in self.constellation.orbits:
            sat_position = orbit.get_satellite_positions()
            all_sat_position.update({orbit.id: sat_position})
            all_orbit_info.update({orbit.id: orbit.get_info})

            if check:
                print(f"orbit info: {orbit.get_info()}")
                for sat in orbit.sats:
                    print(
                        f"sat info of sat-{sat.id}, orbit-{orbit.id} is {sat.get_info()}"
                    )

        self.all_sat_position = all_sat_position  # * position is of ECEF

    def earth_graph(self):
        # ============================================================
        # PyVista scene
        # PyVista provides a textured Earth mesh directly. :contentReference[oaicite:2]{index=2}
        # ============================================================
        earth = examples.planets.load_earth(radius=EARTH_RADIUS)
        earth_texture = examples.load_globe_texture()

        self.plotter = pv.Plotter(off_screen=False, window_size=WINDOW_SIZE)
        self.plotter.set_background("black")

        # Earth
        self.plotter.add_mesh(earth, texture=earth_texture, smooth_shading=True)

        # ============================================================
        # Stars
        # ============================================================
        rng = np.random.default_rng(2026)
        n_stars = 3000
        star_shell_radius = 120000.0

        dirs = rng.normal(size=(n_stars, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        radii = rng.uniform(
            0.90 * star_shell_radius, star_shell_radius, size=(n_stars, 1)
        )
        stars = dirs * radii

        star_cloud = pv.PolyData(stars)
        self.plotter.add_points(
            star_cloud,
            color="white",
            point_size=2,
            render_points_as_spheres=True,
            opacity=0.75,
        )

        # ============================================================
        # Lighting
        # ============================================================
        light1 = pv.Light(
            position=(90000, -30000, 25000),
            focal_point=(0, 0, 0),
            color="white",
            intensity=1.2,
        )
        self.plotter.add_light(light1)

        light2 = pv.Light(
            position=(-50000, 25000, -15000),
            focal_point=(0, 0, 0),
            color="white",
            intensity=0.20,
        )
        self.plotter.add_light(light2)

        # ============================================================
        # Camera
        # Choose a view that shows the constellation structure clearly.
        # ============================================================
        cam_dir = np.array([1.0, -0.8, 0.55], dtype=float)
        cam_dir /= np.linalg.norm(cam_dir)
        cam_dist = 26000.0

        self.plotter.camera.position = tuple(cam_dir * cam_dist)
        self.plotter.camera.focal_point = (0.0, 0.0, 0.0)
        self.plotter.camera.up = (0.0, 0.0, 1.0)
        self.plotter.camera.view_angle = 28.0

        sat_position_matrix = []
        self.const_info_checking(0)  # ! const info check print
        for orbit in self.all_sat_position:
            sat_position_matrix += self.all_sat_position[orbit]
        sat_position_matrix = np.vstack(sat_position_matrix)
        # print(sat_position_matrix)

        orbit_curves = []
        for orbit in self.constellation.orbits:
            nu = np.linspace(0.0, 2.0 * np.pi, 700)
            orbit_pts = []

            R = orbit.sats[0].rotation_matrix
            for u in nu:
                r_orb = np.array(
                    [
                        self.constellation.radius * np.cos(u),
                        self.constellation.radius * np.sin(u),
                        0.0,
                    ]
                )
                orbit_pts.append(R @ r_orb)
            orbit_pts = np.array(orbit_pts)

            orbit_curves.append(orbit_pts)

        for curve in orbit_curves:
            self.plotter.add_lines(curve, color="white", width=0.3, connected=True)

    def show_earth_figure(self):
        self.plotter.show()

    def show_nwk_state_figure(self):
        plt.figure(figsize=(25, 25))
        pos = nx.spring_layout(self.graph)

        self.node_labels = {
            n: f"{n}\nVWQ: {d['sat'].vwq}\nCPU: {d['sat'].fs/1e9}GHz"
            for n, d in self.graph.nodes(data=True)
        }
        nx.draw(
            self.graph,
            pos,
            node_size=10,
            labels=self.node_labels,
            font_size=4,
            with_labels=True,
        )

        self.edge_labels = {
            (u, v): f"{float(d['distance']/1e3):.2f}e3km\n{d['full_bw']/1e6}Mbps"
            for u, v, d in self.graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=self.edge_labels, font_size=4
        )
        plt.show()

    # * update graph: vwq, bw, dist
    def update_graph(self):
        self.constellation.num_isls = 0
        self.constellation.update_constellation(self.delta_t)

        self.earth_graph()
        self.graph = nx.DiGraph()  # * directed graph!

        # Add nodes for each satellite in the constellation
        for orbit in self.constellation.orbits:
            for sat in orbit.sats:
                # Add node with key as (orbit_id, satellite_id) and store the satellite object
                self.graph.add_node((orbit.id, sat.id), sat=sat, fs=sat.fs)

        # Add edges for laser inter-satellite links
        # Intra-plane LISL: add edges between neighboring satellites in the same orbit
        for orbit in self.constellation.orbits:
            if orbit.num_sats < 2:
                continue  # No intra-plane links for single-satellite orbits
            for sat_id in range(1, orbit.num_sats + 1):
                sat1 = (orbit.id, sat_id)
                sat2 = (orbit.id, (sat_id % orbit.num_sats) + 1)

                # * add sat to earth plotter
                self.plotter.add_points(
                    orbit.sats[sat_id - 1].position_ecef,
                    color="red",
                    point_size=7,
                    render_points_as_spheres=True,
                )

                if self._check_isl_feasibility(sat1, sat2):
                    self.constellation.num_isls += 1
                    distance = self.get_distance(sat1, sat2)
                    self.graph.add_edge(
                        sat1,
                        sat2,
                        distance=distance,
                        occupied_bw=self.occupied_bw_vector[(sat1, sat2)],
                        full_bw=self.link_bw,
                    )
                    self.graph.add_edge(
                        sat2,
                        sat1,
                        distance=distance,
                        occupied_bw=self.occupied_bw_vector[(sat2, sat1)],
                        full_bw=self.link_bw,
                    )
                    self.plotter.add_lines(
                        np.array(
                            [
                                orbit.sats[sat_id - 1].position_ecef,
                                orbit.sats[sat_id % orbit.num_sats].position_ecef,
                            ]
                        ),
                        color="red",
                        width=1,
                        connected=True,
                    )
                else:
                    self.bw_vector_del_link(sat1, sat2)

        # Inter-plane LISL: add edges between neighboring satellites in adjacent orbits
        if self.constellation.num_orbits < 2:
            return self.graph  # No inter-plane links for single-orbit constellations
        for orbit_id in range(1, self.constellation.num_orbits + 1):
            if orbit_id != self.constellation.num_orbits:  # ! check point
                # For all orbits except the last one
                for sat_id in range(1, self.constellation.num_sats_per_orbit + 1):
                    sat1 = (orbit_id, sat_id)
                    sat2 = ((orbit_id % self.constellation.num_orbits) + 1, sat_id)
                    if self._check_isl_feasibility(sat1, sat2):
                        self.constellation.num_isls += 1
                        distance = self.get_distance(sat1, sat2)
                        self.graph.add_edge(
                            sat1,
                            sat2,
                            distance=distance,
                            occupied_bw=self.occupied_bw_vector[(sat1, sat2)],
                            full_bw=self.link_bw,
                        )
                        self.graph.add_edge(
                            sat2,
                            sat1,
                            distance=distance,
                            occupied_bw=self.occupied_bw_vector[(sat2, sat1)],
                            full_bw=self.link_bw,
                        )
                        self.plotter.add_lines(
                            np.array(
                                [
                                    self.constellation.orbits[orbit_id - 1]
                                    .sats[sat_id - 1]
                                    .position_ecef,
                                    self.constellation.orbits[
                                        orbit_id % self.constellation.num_orbits
                                    ]
                                    .sats[sat_id - 1]
                                    .position_ecef,
                                ]
                            ),
                            color="black",
                            width=2,
                            connected=True,
                        )
                    else:
                        self.bw_vector_del_link(sat1, sat2)
            else:
                # * For the last orbit, add links to the first orbit, the offset is calculated based on the phasediff. Currently, nothing new is added
                for sat_id in range(1, self.constellation.num_sats_per_orbit + 1):
                    sat1 = (orbit_id, sat_id)
                    sat2 = ((orbit_id % self.constellation.num_orbits) + 1, sat_id)
                    if self._check_isl_feasibility(sat1, sat2):
                        self.constellation.num_isls += 1
                        distance = self.get_distance(sat1, sat2)
                        self.graph.add_edge(
                            sat1,
                            sat2,
                            distance=distance,
                            occupied_bw=self.occupied_bw_vector[(sat1, sat2)],
                            full_bw=self.link_bw,
                        )
                        self.graph.add_edge(
                            sat2,
                            sat1,
                            distance=distance,
                            occupied_bw=self.occupied_bw_vector[(sat2, sat1)],
                            full_bw=self.link_bw,
                        )
                        self.plotter.add_lines(
                            np.array(
                                [
                                    self.constellation.orbits[orbit_id - 1]
                                    .sats[sat_id - 1]
                                    .position_ecef,
                                    self.constellation.orbits[
                                        orbit_id % self.constellation.num_orbits
                                    ]
                                    .sats[sat_id - 1]
                                    .position_ecef,
                                ]
                            ),
                            color="black",
                            width=2,
                            connected=True,
                        )
                    else:
                        self.bw_vector_del_link(sat1, sat2)

    def bw_vector_init(self):
        self.occupied_bw_vector = {}
        self.rand_occupied_bw_vector = {}
        self.nearest_gnd_bw_vector = {}

    def bw_vector_add_link(self, sat1, sat2):
        self.occupied_bw_vector[(sat1, sat2)] = 0
        self.occupied_bw_vector[(sat2, sat1)] = 0

        self.rand_occupied_bw_vector[(sat1, sat2)] = 0
        self.rand_occupied_bw_vector[(sat2, sat1)] = 0

        self.nearest_gnd_bw_vector[(sat1, sat2)] = 0
        self.nearest_gnd_bw_vector[(sat2, sat1)] = 0

    def bw_vector_del_link(self, sat1, sat2):
        self.occupied_bw_vector.pop((sat1, sat2), None)
        self.occupied_bw_vector.pop((sat2, sat1), None)

        self.rand_occupied_bw_vector.pop((sat1, sat2), None)
        self.rand_occupied_bw_vector.pop((sat2, sat1), None)

        self.nearest_gnd_bw_vector.pop((sat1, sat2), None)
        self.nearest_gnd_bw_vector.pop((sat2, sat1), None)

    def _init_graph(self):
        """
        Build the network graph for the Walker Delta Constellation.

        Returns:
        - graph: NetworkX graph representing the satellite network.
        """
        self.earth_graph()
        self.graph = nx.DiGraph()  # * directed graph!
        self.bw_vector_init()
        # Add nodes for each satellite in the constellation
        for orbit in self.constellation.orbits:
            for sat in orbit.sats:
                # Add node with key as (orbit_id, satellite_id) and store the satellite object
                self.graph.add_node((orbit.id, sat.id), sat=sat, fs=sat.fs)

        # Add edges for laser inter-satellite links
        # Intra-plane LISL: add edges between neighboring satellites in the same orbit
        for orbit in self.constellation.orbits:
            if orbit.num_sats < 2:
                continue  # No intra-plane links for single-satellite orbits
            for sat_id in range(1, orbit.num_sats + 1):
                sat1 = (orbit.id, sat_id)
                sat2 = (orbit.id, (sat_id % orbit.num_sats) + 1)

                # * add sat to earth plotter
                self.plotter.add_points(
                    orbit.sats[sat_id - 1].position_ecef,
                    color="red",
                    point_size=7,
                    render_points_as_spheres=True,
                )

                if self._check_isl_feasibility(sat1, sat2):
                    self.constellation.num_isls += 1
                    distance = self.get_distance(sat1, sat2)
                    self.graph.add_edge(
                        sat1,
                        sat2,
                        distance=distance,
                        occupied_bw=0,
                        full_bw=self.link_bw,
                    )
                    self.graph.add_edge(
                        sat2,
                        sat1,
                        distance=distance,
                        occupied_bw=0,
                        full_bw=self.link_bw,
                    )
                    self.plotter.add_lines(
                        np.array(
                            [
                                orbit.sats[sat_id - 1].position_ecef,
                                orbit.sats[sat_id % orbit.num_sats].position_ecef,
                            ]
                        ),
                        color="red",
                        width=1,
                        connected=True,
                    )
                    self.bw_vector_add_link(sat1, sat2)

        # Inter-plane LISL: add edges between neighboring satellites in adjacent orbits
        if self.constellation.num_orbits < 2:
            return self.graph  # No inter-plane links for single-orbit constellations
        for orbit_id in range(1, self.constellation.num_orbits + 1):
            if orbit_id != self.constellation.num_orbits:  # ! check point
                # For all orbits except the last one
                for sat_id in range(1, self.constellation.num_sats_per_orbit + 1):
                    sat1 = (orbit_id, sat_id)
                    sat2 = ((orbit_id % self.constellation.num_orbits) + 1, sat_id)
                    if self._check_isl_feasibility(sat1, sat2):
                        self.constellation.num_isls += 1
                        distance = self.get_distance(sat1, sat2)
                        self.graph.add_edge(
                            sat1,
                            sat2,
                            distance=distance,
                            occupied_bw=0,
                            full_bw=self.link_bw,
                        )
                        self.graph.add_edge(
                            sat2,
                            sat1,
                            distance=distance,
                            occupied_bw=0,
                            full_bw=self.link_bw,
                        )
                        self.plotter.add_lines(
                            np.array(
                                [
                                    self.constellation.orbits[orbit_id - 1]
                                    .sats[sat_id - 1]
                                    .position_ecef,
                                    self.constellation.orbits[
                                        orbit_id % self.constellation.num_orbits
                                    ]
                                    .sats[sat_id - 1]
                                    .position_ecef,
                                ]
                            ),
                            color="black",
                            width=2,
                            connected=True,
                        )
                        self.bw_vector_add_link(sat1, sat2)
            else:
                # * For the last orbit, add links to the first orbit, the offset is calculated based on the phasediff. Currently, nothing new is added
                for sat_id in range(1, self.constellation.num_sats_per_orbit + 1):
                    sat1 = (orbit_id, sat_id)
                    sat2 = ((orbit_id % self.constellation.num_orbits) + 1, sat_id)
                    if self._check_isl_feasibility(sat1, sat2):
                        self.constellation.num_isls += 1
                        distance = self.get_distance(sat1, sat2)
                        # distance = self.get_haversine_distance(sat1, sat2)
                        self.graph.add_edge(
                            sat1,
                            sat2,
                            distance=distance,
                            occupied_bw=0,
                            full_bw=self.link_bw,
                        )
                        self.graph.add_edge(
                            sat2,
                            sat1,
                            distance=distance,
                            occupied_bw=0,
                            full_bw=self.link_bw,
                        )
                        self.plotter.add_lines(
                            np.array(
                                [
                                    self.constellation.orbits[orbit_id - 1]
                                    .sats[sat_id - 1]
                                    .position_ecef,
                                    self.constellation.orbits[
                                        orbit_id % self.constellation.num_orbits
                                    ]
                                    .sats[sat_id - 1]
                                    .position_ecef,
                                ]
                            ),
                            color="black",
                            width=2,
                            connected=True,
                        )
                        self.bw_vector_add_link(sat1, sat2)

    def _check_isl_feasibility(self, vertex_key1, vertex_key2):
        """
        Check if a laser inter-satellite link (LISL) is feasible between two satellites.

        Parameters:
        - vertex_key1: Tuple (orbit_id, sat_id) of the first satellite.
        - vertex_key2: Tuple (orbit_id, sat_id) of the second satellite.

        Returns:
        - True if the LISL is feasible, False otherwise.
        """
        try:
            sat1 = self.graph.nodes[vertex_key1]["sat"]
            sat2 = self.graph.nodes[vertex_key2]["sat"]
        except KeyError:
            raise ValueError("One or both vertex keys do not exist in the network")

        #! current assume all inter/intra-orbital ISL exists
        # mid_point_ecef = (sat1.position_ecef + sat2.position_ecef) / 2
        # earth_radius = EARTH_RADIUS
        # alt_mid_point = np.linalg.norm(mid_point_ecef) - earth_radius - 80
        # if alt_mid_point < 0:
        #     # The LISL is blocked by the Earth
        #     return False

        return True
