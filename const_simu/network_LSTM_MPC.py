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

from NN_module import Predictor_NN, TaskAllocDataset
import torch
from torch import nn
from torch.utils.data import DataLoader

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
                )  #! good! to form a circle loop
                if self._check_isl_feasibility(sat1, sat2):
                    distance = self.get_distance(sat1, sat2)

                    self.graph.add_edge(
                        sat1, sat2, weight=distance
                    )  #! maybe change the attribute

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
                    )  #! also, distance-based link establishment could be added
                    self.graph.add_edge(sat1, sat2, weight=distance)

        return self.graph  #! important fault!

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
        )  #! 80 km residue
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
            if orbit_id != self.constellation.num_orbits:  #! check point
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
                    sat_id_offset = round(sat_id_offset)  # Round to the nearest integer
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
        N_max,
        a_max,
        task_workload,
        link_bw,
        delta_t,
        max_time_span,
        train_time_unit_len,
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
        self.link_bw = link_bw
        self.N_max = N_max
        self.a_max = a_max
        self.task_workload = task_workload
        self.delta_t = delta_t
        self.max_time_span = max_time_span
        self.train_time_unit_len = train_time_unit_len
        self.bw_min = bw_min
        self.bw_max = bw_max

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
        self.T_pred = self.T_mem
        print(f"Delta: {self.delta_t}, T memory: {self.T_mem}")

        self.T_train_data = train_time_unit_len * (self.T_mem + self.T_pred)
        # print(f"the time will be: {self.T_train_data}")

        self.V = 10
        self.m1 = 1 / self.constellation.num_isls if self.constellation.num_isls else 0
        self.m2 = 1 / self.constellation.num_sats
        self.m3 = 0.1 / (self.constellation.radius * self.a_max)
        self.m4 = 1

        self.vwq_matrix = np.zeros(
            [self.constellation.num_orbits, self.constellation.num_sats_per_orbit]
        )
        self.get_constellation_haversine_matrix()
        self._init_graph()

        with open("check_point/organized_train_data.txt", "w") as f:
            pass
        with open("check_point/predicted_data.json", "w") as f:
            pass
        with open("check_point/vwq.txt", "w") as f:
            pass
        with open("check_point/occupied_bw.json", "w") as f:
            pass

    def simu_task_arriving(self, task_num):
        """
        generate poisson arrivals for each task (with maximum truncate)
        """
        # * select random task arrivals
        selected_sat_indecies = self.rng.integers(
            self.constellation.num_sats, size=task_num
        )  # random pick satellites
        task_size_vector = abs(
            self.rng.normal(self.a_max, size=task_num)
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
                )
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
            ]  # * task[0]: (s_orbit, s_sat), task[1]: computational size = task size * task workload
        )

        self.haver_task = cp.sum(
            [  #! haversine weighed offloading
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
        r1 = np.asarray(sat1.position_ecef, dtype=float)
        r2 = np.asarray(sat2.position_ecef, dtype=float)

        cos_theta = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return sat1.radius * theta

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

        print(f"Result: {self.cvx_full_prob.solve()}")
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
                    )  #! check point

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
                    )  #! check point

            for orbit_id in range(len(self.constellation.orbits)):
                orbit = self.constellation.orbits[orbit_id]
                for sat_id in range(len(orbit.sats)):
                    sat = orbit.sats[sat_id]
                    sat.vwq = max(0, sat.vwq - sat.fs)
                    self.vwq_matrix[orbit_id][sat_id] = sat.vwq

        self.new_task_alloc.append(temp_arrivals)

    def parse_train_data(self, local_data=0, save_train_data=0):
        if local_data:
            self.new_task_alloc = np.load("new_task_alloc.npy")

        # * extract full train data: input & output
        self.organized_train_data = []
        for i in range(len(self.new_task_alloc)):
            temp_data = []
            for j in range(i, i + self.T_pred + self.T_mem):
                if j < len(self.new_task_alloc):
                    temp_data.append(self.new_task_alloc[j])
            if len(temp_data) == self.T_pred + self.T_mem:
                self.organized_train_data.append(temp_data)

        # ! check point
        # print(f"this is the trained data\n{self.organized_train_data}")
        with open("check_point/organized_train_data.txt", "w") as f:
            f.write(f"{self.organized_train_data}")

        if save_train_data:
            np.save("train_data.npy", self.organized_train_data)

        temp_data = np.asarray(self.organized_train_data, dtype=np.float32)

        (
            num_samples,
            total_len,
            orbit_num,
            sat_num_per_orbit,
            _,
            _,
        ) = temp_data.shape
        X = temp_data[:, : self.T_mem, :, :, :, :]
        Y = temp_data[:, self.T_mem :, :, :, :, :]

        # * flatten the last dimension
        X = X.reshape(num_samples, self.T_mem, (orbit_num * sat_num_per_orbit) ** 2)
        Y = Y.reshape(num_samples, self.T_pred, (orbit_num * sat_num_per_orbit) ** 2)
        return X, Y

    def prepare_train_test_dataset(self, X, Y, batch_size):
        num_train = int(self.train_ratio * X.shape[0])
        X_train = X[:num_train]
        Y_train = Y[:num_train]
        X_test = X[num_train:]
        Y_test = Y[num_train:]

        self.train_dataset = TaskAllocDataset(X_train, Y_train)
        self.test_dataset = TaskAllocDataset(X_test, Y_test)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size, shuffle=True)

    def train_predictor(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x_b, y_b in self.train_dataloader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)

                self.optimizer.zero_grad()  # clear gradients for this training step
                pred = self.model(x_b)
                loss = self.criterion(pred, y_b)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * x_b.size(0)

            if not (epoch % 50):
                epoch_loss /= len(self.train_dataset)
                print(f"Epoch [{epoch}/{epochs}] Average Loss:{epoch_loss:.5f}")

    def prepare_predictor(
        self, epochs=100, batch_size=16, hidden_size=40, lr=1e-4, num_layers=2
    ):
        """run the full time-span simulation"""
        self.rng = np.random.default_rng(seed=self.train_seed)

        # * num of tasks arrived each slot
        self.task_number_vector = self.rng.poisson(
            self.N_max, size=int(self.T_train_data)
        )

        for t in self.task_number_vector:
            # * new service request generation to each sat
            self.simu_task_arriving(t)

            # * solve the task assignment problem
            self.task_assignment_subproblem()

            # * update: vwq & sat_nwk dynamics
            self.vwq_update(check=0)
            self.update_graph()

        # * save train data collection
        # print(self.new_task_alloc)
        # np.save("new_task_alloc.npy", self.new_task_alloc)

        # * organize data for training
        X, Y = self.parse_train_data(save_train_data=0)

        # * build dataset & dataloader
        self.train_ratio = 0.8
        self.prepare_train_test_dataset(X, Y, batch_size)

        input_size = X.shape[-1]
        output_size = Y.shape[-1]

        # * set up the nwk, criterion, optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Predictor_NN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            pred_len=self.T_pred,
            num_layers=num_layers,
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # * train the nwk
        # epochs: training times on the entire training set
        # batch: the size of data processed each time
        self.train_predictor(epochs)

        # * test the nwk
        self.test_predictor()

    def graph_reset(self):
        self.constellation = self.original_constellation
        self.vwq_matrix = np.zeros(
            [self.constellation.num_orbits, self.constellation.num_sats_per_orbit]
        )
        self._init_graph()

    def test_predictor(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for xb, yb in self.test_dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                test_loss += loss.item() * xb.size(0)
        test_loss /= len(self.test_dataset)
        print(f"Average test loss: {test_loss}")

    def full_simu(self, using_learning=0):
        """run the full time-span simulation"""
        #! remember to reinitialize the system fully from the training state
        self.graph_reset()

        self.rng = np.random.default_rng(
            seed=self.run_seed
        )  #! train with seed = 64, predict with seed = 102
        self.task_number_vector = self.rng.poisson(
            self.N_max, size=int(self.max_time_span)
        )

        #! test task data
        # self.task_number_vector = self.rng.poisson(
        #     self.N_max,
        #     size=1,
        # )
        # self.task_number_vector = np.append(self.task_number_vector, 0)
        # self.task_number_vector = np.append(self.task_number_vector, 0)
        # self.task_number_vector = np.append(self.task_number_vector, 0)
        # print(f"task data: {self.task_number_vector}")

        self.new_task_alloc = []
        self.occupied_full_history = []
        if using_learning:
            self.model.eval()
            for i in range(self.T_mem - 1):
                self.new_task_alloc.append(
                    np.zeros(
                        [
                            self.constellation.num_orbits,
                            self.constellation.num_sats_per_orbit,
                            self.constellation.num_orbits,
                            self.constellation.num_sats_per_orbit,
                        ]
                    )
                )

        for i in range(len(self.task_number_vector)):
            print(f":::::::::::::::::::THIS IS SLOT {i}::::::::::::::::::::")
            # * new service request generation to each sat
            self.simu_task_arriving(self.task_number_vector[i])

            # * solve the task assignment problem
            self.task_assignment_subproblem()

            # * update vwq
            self.vwq_update(check=1)

            # * solve the MPC communication problem
            self.solve_MPC_comm(i)

            # * update: bw occupation & sat_nwk dynamics
            self.history_update()

            self.update_graph()  # update the nwk graph

        with open("check_point/full_history.json", "w") as f:
            f.write(f"{self.occupied_full_history}")

    def solve_MPC_comm(self, i, using_learning=0):
        if using_learning:
            with torch.no_grad():
                # * gather data, shape: [T_mem, num_orbits, num_sat_per_orbit]
                x_mem = np.array(self.new_task_alloc[i : i + self.T_mem]).reshape(
                    self.T_mem,
                    (
                        self.constellation.num_orbits
                        * self.constellation.num_sats_per_orbit
                    )
                    ** 2,
                )

                # * torch must use torch.tensor datatype; also, since the forward in model is a batch-based, a new batch dim should be inserted at the beginning
                x_mem = torch.tensor(
                    x_mem, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                # * model predict
                future_alloc = self.model(x_mem).flatten()
                for i in range(len(future_alloc)):
                    future_alloc[i] = (future_alloc[i] > 1e-3) * future_alloc[i]

            future_alloc = future_alloc.reshape(
                x_mem.size(0),
                self.T_pred,
                self.constellation.num_orbits,
                self.constellation.num_sats_per_orbit,
                self.constellation.num_orbits,
                self.constellation.num_sats_per_orbit,
            )

        else:  # * duplicate the last slot data padding the future prediction
            self.future_alloc = [copy.deepcopy(self.new_task_alloc[-1])]
            for x in reversed(self.new_task_alloc):
                if x:
                    break
            for _ in range(self.T_pred - 1):
                self.future_alloc.append(copy.deepcopy(x))

        # * set the decision var: routing & bw
        self.cvx_create_MPC_comm_vars()

        # * set the obj func
        self.cvx_create_MPC_obj()

        # * set the constr: 1. min, max bw 2. routing regulation 3. time-length bw
        self.cvx_create_MPC_constraints(i)

        # * set the problem
        self.cvx_MPC_full_prob = cp.Problem(self.cvx_MPC_obj, self.cvx_MPC_constraints)

        # * solve
        print(f"MPC RESULT IS HERE: {self.cvx_MPC_full_prob.solve(verbose=False)}")

    def history_update(self):
        history_leaving = {}
        new_arriving = {}
        for s, d in self.graph.edges():
            # ! redundant checking for occupied_bw_vector
            # B[t+1] = B[t] + a(t) - b(t)
            # history_leaving[(s, d)] = 0
            # new_arriving[(s, d)] = 0

            # for sat in self.graph.nodes():
            #     for task_id in range(len(self.cvx_MPC_comm_data[0])):
            #         new_arriving[(s, d)] += self.cvx_MPC_comm_data[0][task_id]["var"][
            #             "auxiliary_bw_rt_bilinear"
            #         ][sat][(s, d)].value

            #     for t in range(1, 1 + len(self.occupied_full_history)):
            #         for task_id in range(len(self.occupied_full_history[-t])):
            #             history_leaving[(s, d)] += (
            #                 self.occupied_full_history[-t][task_id]["var"]["bw"][sat]
            #                 * self.occupied_full_history[-t][task_id]["var"]["routing"][
            #                     sat
            #                 ][(s, d)]["selected"]
            #                 * (
            #                     self.occupied_full_history[-t][task_id]["var"][
            #                         "routing"
            #                     ][sat][(s, d)]["trans_time"]
            #                     == t
            #                 )
            #             )

            # print(
            #     f"coming: {new_arriving[(s, d)]} leaving:{history_leaving[(s, d)]} handy calculate: {self.occupied_bw_vector[(s, d)] + new_arriving[(s, d)] - history_leaving[(s, d)]} existing bw: {self.B_evolving[1][(s, d)].value}"
            # )

            self.occupied_bw_vector[(s, d)] = self.B_evolving[1][(s, d)].value

        self.occupied_full_history.append([])
        # * only apply the first solution result
        # list of dictionaries
        for idx, dt in enumerate(self.future_alloc[0]):
            for key, val in dt.items():
                task_description = key
                task_data = val
            self.occupied_full_history[-1].append(
                {
                    "task_description": task_description,
                    "task_data": task_data,
                    "var": {"bw": {}, "routing": {}, "trans_time": {}},
                }
            )
            for sat in self.graph.nodes():
                s_orbit = sat[0]
                s_sat = sat[1]
                temp_bw = self.cvx_MPC_comm_data[0][idx]["var"]["bw"][sat].value
                if temp_bw > 1e-3:
                    self.occupied_full_history[-1][idx]["var"]["bw"][sat] = temp_bw
                    self.occupied_full_history[-1][idx]["var"]["routing"][sat] = {}
                    for s, d in self.graph.edges():
                        self.occupied_full_history[-1][idx]["var"]["routing"][sat][
                            (s, d)
                        ] = {
                            "selected": self.cvx_MPC_comm_data[0][idx]["var"][
                                "routing"
                            ][sat][(s, d)].value,
                            "trans_time": np.ceil(
                                task_data[s_orbit - 1][s_sat - 1] / temp_bw
                                + self.max_propagation_delay
                            )
                            * self.cvx_MPC_comm_data[0][idx]["var"]["routing"][sat][
                                (s, d)
                            ].value,
                        }
                else:
                    self.occupied_full_history[-1][idx]["var"]["bw"][sat] = 0
                    self.occupied_full_history[-1][idx]["var"]["routing"][sat] = {}
                    for s, d in self.graph.edges():
                        self.occupied_full_history[-1][idx]["var"]["routing"][sat][
                            (s, d)
                        ] = {"selected": 0, "trans_time": 0}

        if len(self.occupied_full_history) > (self.T_mem + 2):
            self.occupied_full_history.pop(0)

        print(f"occupied bw:\n{self.occupied_bw_vector}\n")
        with open("check_point/occupied_bw.json", "a") as f:
            f.write(f"{self.occupied_bw_vector}")

    def cvx_create_MPC_constraints_original(self, time_step):
        self.cvx_MPC_constraints = []
        self.B_evolving = [{}]
        history_leaving = []
        all_arrivals = []

        for t in range(self.T_pred):
            for task_id in range(len(self.cvx_MPC_comm_data[t])):
                src_sat = (
                    self.cvx_MPC_comm_data[t][task_id]["task_description"][0],
                    self.cvx_MPC_comm_data[t][task_id]["task_description"][1],
                )
                task_data = self.cvx_MPC_comm_data[t][task_id]["task_data"]

                for dst_sat in self.graph.nodes():  # * this loop for offloading dst
                    #! routing regulation
                    if (task_data[dst_sat[0] - 1][dst_sat[1] - 1] > 1e-4) and (
                        dst_sat != src_sat
                    ):  # * if need offload
                        # min, max bw
                        self.cvx_MPC_constraints += [
                            self.cvx_MPC_comm_data[t][task_id]["var"]["bw"][dst_sat]
                            <= self.bw_max
                        ]
                        self.cvx_MPC_constraints += [
                            self.cvx_MPC_comm_data[t][task_id]["var"]["bw"][dst_sat]
                            >= self.bw_min
                        ]

                        for (
                            sat_in_the_route
                        ) in self.graph.nodes():  # * this loop for routing
                            if sat_in_the_route == src_sat:  # source regulation
                                self.cvx_MPC_constraints += [
                                    cp.sum(
                                        [
                                            self.cvx_MPC_comm_data[t][task_id]["var"][
                                                "routing"
                                            ][dst_sat][(sat_in_the_route, neighbor)]
                                            - self.cvx_MPC_comm_data[t][task_id]["var"][
                                                "routing"
                                            ][dst_sat][(neighbor, sat_in_the_route)]
                                            for neighbor in self.graph[sat_in_the_route]
                                        ]
                                    )
                                    == 1
                                ]
                            elif sat_in_the_route == dst_sat:  # dst regulation
                                self.cvx_MPC_constraints += [
                                    cp.sum(
                                        [
                                            self.cvx_MPC_comm_data[t][task_id]["var"][
                                                "routing"
                                            ][dst_sat][(sat_in_the_route, neighbor)]
                                            - self.cvx_MPC_comm_data[t][task_id]["var"][
                                                "routing"
                                            ][dst_sat][(neighbor, sat_in_the_route)]
                                            for neighbor in self.graph[sat_in_the_route]
                                        ]
                                    )
                                    == -1
                                ]
                            else:  # mid regulation
                                self.cvx_MPC_constraints += [
                                    cp.sum(
                                        [
                                            self.cvx_MPC_comm_data[t][task_id]["var"][
                                                "routing"
                                            ][dst_sat][(sat_in_the_route, neighbor)]
                                            - self.cvx_MPC_comm_data[t][task_id]["var"][
                                                "routing"
                                            ][dst_sat][(neighbor, sat_in_the_route)]
                                            for neighbor in self.graph[sat_in_the_route]
                                        ]
                                    )
                                    == 0
                                ]

                    else:  # no offloading/self-offloading: no bw, no routing
                        self.cvx_MPC_constraints += [
                            self.cvx_MPC_comm_data[t][task_id]["var"]["bw"][dst_sat]
                            == 0
                        ]
                        for sat_in_the_route in self.graph.nodes():
                            self.cvx_MPC_constraints += [
                                self.cvx_MPC_comm_data[t][task_id]["var"]["routing"][
                                    dst_sat
                                ][(sat_in_the_route, neighbor)]
                                == 0
                                for neighbor in self.graph[sat_in_the_route]
                            ]

            #! bw evolution
            all_arrivals.append({})
            history_leaving.append({})
            for s, d in self.graph.edges():
                all_arrivals[t][(s, d)] = 0
                history_leaving[t][(s, d)] = 0

                # * calculate history leavings
                for tau in range(self.T_mem):  # only the past arrivals is related
                    # history += task released at t-tau
                    mem_t = t + time_step - (tau + 1)
                    if mem_t >= 0:
                        if mem_t >= time_step:
                            future_t = mem_t - time_step
                            temp_history = cp.hstack(
                                [
                                    self.cvx_MPC_comm_data[future_t][task_id]["var"][
                                        "routing"
                                    ][dst_sat][(s, d)]
                                    * self.cvx_MPC_comm_data[future_t][task_id]["var"][
                                        "bw"
                                    ][dst_sat]
                                    * (
                                        (
                                            self.cvx_MPC_comm_data[future_t][task_id][
                                                "var"
                                            ]["trans_time"][dst_sat]
                                            + future_t
                                        )
                                        == t
                                    )
                                    #! remember to dump duplications
                                    # * the_time_it's_released
                                    for dst_sat in self.graph.nodes()
                                    for task_id in range(len(self.cvx_MPC_comm_data[t]))
                                ]
                            )
                        else:
                            temp_history = cp.hstack(
                                [
                                    self.occupied_full_history[mem_t][task_id]["var"][
                                        "routing"
                                    ][dst_sat][(s, d)]["selected"]
                                    * self.occupied_full_history[mem_t][task_id]["var"][
                                        "bw"
                                    ][dst_sat]
                                    * (
                                        (
                                            self.occupied_full_history[mem_t][task_id][
                                                "var"
                                            ]["routing"][dst_sat][(s, d)]["trans_time"]
                                            + mem_t
                                        )
                                        == t
                                    )
                                    # * the_time_it's_released
                                    for dst_sat in self.graph.nodes()
                                    for task_id in range(len(self.cvx_MPC_comm_data[t]))
                                ]
                            )
                        history_leaving[t][(s, d)] += cp.sum(temp_history)

                # * calculate current arrivals
                all_arrivals[t][(s, d)] = cp.sum(
                    [
                        self.cvx_MPC_comm_data[t][task_id]["var"]["bw"][dst_sat]
                        * self.cvx_MPC_comm_data[t][task_id]["var"]["routing"][dst_sat][
                            (s, d)
                        ]
                        for dst_sat in self.graph.nodes()
                        for task_id in range(len(self.cvx_MPC_comm_data[t]))
                    ]
                )

            self.B_evolving.append({})
            for s, d in self.graph.edges():
                if t == 0:
                    self.B_evolving[t][(s, d)] = copy.deepcopy(
                        self.occupied_bw_vector[(s, d)]
                    )

                # self.B_evolving[t+1] = self.B_evolving[t] + new_arr[t] - old_release[t]
                self.B_evolving[t + 1][(s, d)] = self.B_evolving[t][(s, d)] + (
                    all_arrivals[t][(s, d)] - history_leaving[t][(s, d)]
                )
                self.cvx_MPC_constraints += [
                    self.B_evolving[t + 1][(s, d)] <= self.graph[s][d]["full_bw"]
                ]

    def cvx_create_MPC_constraints(self, time_step):
        self.cvx_MPC_constraints = []
        self.B_evolving = [{}]
        history_leaving = []
        all_arrivals = []

        for t in range(self.T_pred):
            for task_id in range(len(self.cvx_MPC_comm_data[t])):
                src_sat = (
                    int(self.cvx_MPC_comm_data[t][task_id]["task_description"][0][0])
                    + 1,
                    int(self.cvx_MPC_comm_data[t][task_id]["task_description"][0][1])
                    + 1,
                )
                task_data = self.cvx_MPC_comm_data[t][task_id]["task_data"]

                for dst_sat in self.graph.nodes():  # * this loop for offloading dst
                    #! routing regulation
                    if (task_data[dst_sat[0] - 1][dst_sat[1] - 1] > 1e-4) and (
                        dst_sat != src_sat
                    ):  # * if need offload
                        # min, max bw
                        self.cvx_MPC_constraints += [
                            self.cvx_MPC_comm_data[t][task_id]["var"]["bw"][dst_sat]
                            <= self.bw_max
                        ]
                        self.cvx_MPC_constraints += [
                            self.cvx_MPC_comm_data[t][task_id]["var"]["bw"][dst_sat]
                            >= self.bw_min
                        ]

                        for (
                            sat_in_the_route
                        ) in self.graph.nodes():  # * this loop for routing
                            if sat_in_the_route == src_sat:  # source regulation
                                self.cvx_MPC_constraints += [
                                    cp.sum(
                                        [
                                            self.cvx_MPC_comm_data[t][task_id]["var"][
                                                "routing"
                                            ][dst_sat][(sat_in_the_route, neighbor)]
                                            for neighbor in self.graph[sat_in_the_route]
                                        ]
                                    )
                                    == 1
                                ] + [
                                    cp.sum(
                                        [
                                            self.cvx_MPC_comm_data[t][task_id]["var"][
                                                "routing"
                                            ][dst_sat][(neighbor, sat_in_the_route)]
                                            for neighbor in self.graph[sat_in_the_route]
                                        ]
                                    )
                                    == 0
                                ]
                            elif sat_in_the_route == dst_sat:  # dst regulation
                                self.cvx_MPC_constraints += [
                                    cp.sum(
                                        [
                                            self.cvx_MPC_comm_data[t][task_id]["var"][
                                                "routing"
                                            ][dst_sat][(sat_in_the_route, neighbor)]
                                            for neighbor in self.graph[sat_in_the_route]
                                        ]
                                    )
                                    == 0
                                ] + [
                                    cp.sum(
                                        [
                                            self.cvx_MPC_comm_data[t][task_id]["var"][
                                                "routing"
                                            ][dst_sat][(neighbor, sat_in_the_route)]
                                            for neighbor in self.graph[sat_in_the_route]
                                        ]
                                    )
                                    == 1
                                ]
                            else:  # mid regulation
                                self.cvx_MPC_constraints += [
                                    cp.sum(
                                        [
                                            self.cvx_MPC_comm_data[t][task_id]["var"][
                                                "routing"
                                            ][dst_sat][(sat_in_the_route, neighbor)]
                                            - self.cvx_MPC_comm_data[t][task_id]["var"][
                                                "routing"
                                            ][dst_sat][(neighbor, sat_in_the_route)]
                                            for neighbor in self.graph[sat_in_the_route]
                                        ]
                                    )
                                    == 0
                                ]

                        #! 4constraints for auxiliary bilnear term
                        for s, d in self.graph.edges():
                            self.cvx_MPC_constraints += (
                                [
                                    self.cvx_MPC_comm_data[t][task_id]["var"][
                                        "auxiliary_bw_rt_bilinear"
                                    ][dst_sat][(s, d)]
                                    >= self.bw_min
                                    * self.cvx_MPC_comm_data[t][task_id]["var"][
                                        "routing"
                                    ][dst_sat][(s, d)]
                                ]
                                + [
                                    self.cvx_MPC_comm_data[t][task_id]["var"][
                                        "auxiliary_bw_rt_bilinear"
                                    ][dst_sat][(s, d)]
                                    >= self.bw_max
                                    * self.cvx_MPC_comm_data[t][task_id]["var"][
                                        "routing"
                                    ][dst_sat][(s, d)]
                                    + self.cvx_MPC_comm_data[t][task_id]["var"]["bw"][
                                        dst_sat
                                    ]
                                    - self.bw_max
                                ]
                                + [
                                    self.cvx_MPC_comm_data[t][task_id]["var"][
                                        "auxiliary_bw_rt_bilinear"
                                    ][dst_sat][(s, d)]
                                    <= self.bw_max
                                    * self.cvx_MPC_comm_data[t][task_id]["var"][
                                        "routing"
                                    ][dst_sat][(s, d)]
                                ]
                                + [
                                    self.cvx_MPC_comm_data[t][task_id]["var"][
                                        "auxiliary_bw_rt_bilinear"
                                    ][dst_sat][(s, d)]
                                    <= self.bw_min
                                    * self.cvx_MPC_comm_data[t][task_id]["var"][
                                        "routing"
                                    ][dst_sat][(s, d)]
                                    + self.cvx_MPC_comm_data[t][task_id]["var"]["bw"][
                                        dst_sat
                                    ]
                                    - self.bw_min
                                ]
                            )

                    else:  # no offloading/self-offloading: no bw, no routing
                        self.cvx_MPC_constraints += [
                            self.cvx_MPC_comm_data[t][task_id]["var"]["bw"][dst_sat]
                            == 0
                        ]
                        for sat_in_the_route in self.graph.nodes():
                            self.cvx_MPC_constraints += [
                                self.cvx_MPC_comm_data[t][task_id]["var"]["routing"][
                                    dst_sat
                                ][(sat_in_the_route, neighbor)]
                                == 0
                                for neighbor in self.graph[sat_in_the_route]
                            ]
                        for s, d in self.graph.edges():
                            self.cvx_MPC_constraints += [
                                self.cvx_MPC_comm_data[t][task_id]["var"][
                                    "auxiliary_bw_rt_bilinear"
                                ][dst_sat][(s, d)]
                                == 0
                            ]

            #! bw evolution
            all_arrivals.append({})
            history_leaving.append({})
            for s, d in self.graph.edges():

                all_arrivals[t][(s, d)] = 0
                history_leaving[t][(s, d)] = 0

                # * calculate history leavings
                for tau in range(self.T_mem):  # only the past arrivals is related
                    # history += task released at t-tau
                    mem_t = t + time_step - (tau + 1)
                    if mem_t >= 0:
                        if mem_t >= time_step:
                            future_t = mem_t - time_step
                            temp_history = [
                                self.cvx_MPC_comm_data[future_t][task_id]["var"][
                                    "auxiliary_bw_rt_bilinear"
                                ][dst_sat][(s, d)]
                                * (
                                    (
                                        self.cvx_MPC_comm_data[future_t][task_id][
                                            "var"
                                        ]["trans_time"][dst_sat]
                                        + future_t
                                    )
                                    == t
                                )
                                #! remember to dump duplications
                                # * the_time_it's_released
                                for dst_sat in self.graph.nodes()
                                for task_id in range(
                                    len(self.cvx_MPC_comm_data[future_t])
                                )
                            ]
                            temp_history = (
                                cp.hstack(temp_history)
                                if temp_history
                                else temp_history
                            )
                        else:
                            # temp_history = cp.hstack(
                            temp_history = np.sum(
                                [
                                    self.occupied_full_history[mem_t][task_id]["var"][
                                        "routing"
                                    ][dst_sat][(s, d)]["selected"]
                                    * self.occupied_full_history[mem_t][task_id]["var"][
                                        "bw"
                                    ][dst_sat]
                                    * (
                                        (
                                            self.occupied_full_history[mem_t][task_id][
                                                "var"
                                            ]["routing"][dst_sat][(s, d)]["trans_time"]
                                            + mem_t
                                        )
                                        == t + time_step
                                    )
                                    # * the_time_it's_released
                                    #! need to decrease the accumulation range
                                    for dst_sat in self.graph.nodes()
                                    for task_id in range(
                                        len(self.occupied_full_history[mem_t])
                                    )
                                ]
                            )
                        history_leaving[t][(s, d)] += cp.sum(temp_history)

                # * calculate current arrivals
                all_arrivals[t][(s, d)] = cp.sum(
                    [
                        self.cvx_MPC_comm_data[t][task_id]["var"][
                            "auxiliary_bw_rt_bilinear"
                        ][dst_sat][(s, d)]
                        for dst_sat in self.graph.nodes()
                        for task_id in range(len(self.cvx_MPC_comm_data[t]))
                    ]
                )

            self.B_evolving.append({})
            for s, d in self.graph.edges():
                if t == 0:
                    self.B_evolving[t][(s, d)] = copy.deepcopy(
                        self.occupied_bw_vector[(s, d)]
                    )

                # self.B_evolving[t] = self.B_evolving[t-1] + new_arr[t] - old_release[t]
                self.B_evolving[t + 1][(s, d)] = self.B_evolving[t][(s, d)] + (
                    all_arrivals[t][(s, d)] - history_leaving[t][(s, d)]
                )
                self.cvx_MPC_constraints += [
                    self.B_evolving[t + 1][(s, d)] <= self.graph[s][d]["full_bw"]
                ]

        # print(f"MPC CONNSTRAINTS:\n{self.cvx_MPC_constraints}")

    def cvx_create_MPC_obj(self):
        self.link_load_vector = {}
        self.route_len = {}

        # calculate load for each link 1-by-1
        for s, d in self.graph.edges():
            self.link_load_vector[(s, d)] = 0
            self.route_len[(s, d)] = 0
            for t in range(self.T_pred):
                for task_id in range(len(self.cvx_MPC_comm_data[t])):
                    for sat in self.graph.nodes():
                        self.link_load_vector[(s, d)] += (
                            self.cvx_MPC_comm_data[t][task_id]["var"][
                                "auxiliary_bw_rt_bilinear"
                            ][sat][(s, d)]
                            / self.graph[s][d]["full_bw"]
                        )
                        self.route_len[(s, d)] += self.cvx_MPC_comm_data[t][task_id][
                            "var"
                        ]["routing"][sat][(s, d)]

        #! check point
        # print(
        #     f"bw:\n{self.graph[s][d]["full_bw"]}\nOBJ to be summed:\n{self.link_load_vector}"
        # )
        loads = cp.hstack(
            [self.link_load_vector[(s, d)] for s, d in self.graph.edges()]
        )
        route_len = cp.hstack([self.route_len[(s, d)] for s, d in self.graph.edges()])
        self.cvx_MPC_obj = cp.Minimize(cp.sum_squares(loads) + cp.sum(route_len))

    def cvx_create_MPC_comm_vars(self):
        self.cvx_MPC_comm_data = []
        for t in range(self.T_pred):
            self.cvx_MPC_comm_data.append([])
            for idx, dt in enumerate(self.future_alloc[t]):
                for key, val in dt.items():
                    task_description = key
                    task_data = val
                self.cvx_MPC_comm_data[t].append(
                    {
                        "task_description": task_description,
                        "task_data": task_data,
                        "var": {
                            "bw": {},
                            "trans_time": {},
                            "routing": {},
                            "auxiliary_bw_rt_bilinear": {},
                        },
                    }
                )
                for sat in self.graph.nodes():
                    # bw var
                    self.cvx_MPC_comm_data[t][idx]["var"]["bw"][sat] = cp.Variable()

                    # * scaling trans_time
                    s_orbit = sat[0] - 1
                    s_sat = sat[1] - 1
                    temp_data = task_data[s_orbit][s_sat]
                    src_orbit = int(task_description[0][0])
                    src_sat = int(task_description[0][1])
                    if ((src_orbit, src_sat) != (s_orbit, s_sat)) and (
                        temp_data > 1e-3
                    ):
                        self.cvx_MPC_comm_data[t][idx]["var"]["trans_time"][sat] = (
                            self.max_propagation_delay + temp_data / self.bw_min
                        )
                    else:
                        self.cvx_MPC_comm_data[t][idx]["var"]["trans_time"][sat] = 0

                    # routing var
                    self.cvx_MPC_comm_data[t][idx]["var"]["routing"][sat] = {}
                    self.cvx_MPC_comm_data[t][idx]["var"]["auxiliary_bw_rt_bilinear"][
                        sat
                    ] = {}
                    for s, d in self.graph.edges():
                        self.cvx_MPC_comm_data[t][idx]["var"]["routing"][sat][
                            (s, d)
                        ] = cp.Variable(boolean=True)
                        self.cvx_MPC_comm_data[t][idx]["var"][
                            "auxiliary_bw_rt_bilinear"
                        ][sat][(s, d)] = cp.Variable()

                # print(f"cvx_MPC_var: {self.cvx_MPC_comm_data[t]}")

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
        self.const_info_checking(0)  #! const info check print
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
        self.constellation.update_constellation(self.delta_t)

        self.earth_graph()
        self.graph = nx.DiGraph()  # * directed graph!

        # Add nodes for each satellite in the constellation
        for orbit in self.constellation.orbits:
            for sat in orbit.sats:
                # Add node with key as (orbit_id, satellite_id) and store the satellite object
                self.graph.add_node((orbit.id, sat.id), sat=sat)

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
                    self.occupied_bw_vector.pop((sat1, sat2), None)
                    self.occupied_bw_vector.pop((sat2, sat1), None)

        # Inter-plane LISL: add edges between neighboring satellites in adjacent orbits
        if self.constellation.num_orbits < 2:
            return self.graph  # No inter-plane links for single-orbit constellations
        for orbit_id in range(1, self.constellation.num_orbits + 1):
            if orbit_id != self.constellation.num_orbits:  #! check point
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
                        self.occupied_bw_vector.pop((sat1, sat2), None)
                        self.occupied_bw_vector.pop((sat2, sat1), None)
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
                        self.occupied_bw_vector.pop((sat1, sat2), None)
                        self.occupied_bw_vector.pop((sat2, sat1), None)

    def _init_graph(self):
        """
        Build the network graph for the Walker Delta Constellation.

        Returns:
        - graph: NetworkX graph representing the satellite network.
        """
        self.earth_graph()
        self.graph = nx.DiGraph()  # * directed graph!
        self.occupied_bw_vector = {}
        # Add nodes for each satellite in the constellation
        for orbit in self.constellation.orbits:
            for sat in orbit.sats:
                # Add node with key as (orbit_id, satellite_id) and store the satellite object
                self.graph.add_node((orbit.id, sat.id), sat=sat)

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
                    self.occupied_bw_vector[(sat1, sat2)] = 0
                    self.occupied_bw_vector[(sat2, sat1)] = 0

        # Inter-plane LISL: add edges between neighboring satellites in adjacent orbits
        if self.constellation.num_orbits < 2:
            return self.graph  # No inter-plane links for single-orbit constellations
        for orbit_id in range(1, self.constellation.num_orbits + 1):
            if orbit_id != self.constellation.num_orbits:  #! check point
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
                        self.occupied_bw_vector[(sat1, sat2)] = 0
                        self.occupied_bw_vector[(sat2, sat1)] = 0
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
                        self.occupied_bw_vector[(sat1, sat2)] = 0
                        self.occupied_bw_vector[(sat2, sat1)] = 0

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
