import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

EARTH_RADIUS = 6378.137  # km

# * per sate arrivals
N_max = 12  # max task number within the constellation
a_max = 100  # M task size
task_workload = 3e2  # cyc/bit
link_bw = 1e4  # Mbps per link. 20G?
fs = 1e4  # M Hz sat CPU freq
delta_t = (
    1  # slot length #! in the current nwk implementation, the delta_t is taken as 1
)
max_time_span = 20  # slot num

bw_min = 20  # Mbps for a task
bw_max = 15  # Mbps for a task

# * constellation parameter setting
N_const = 7
num_orbits = N_const
num_sats_per_orbit = N_const
altitude = 1050
radius = altitude + EARTH_RADIUS
inclination = 53.0
phasediff = 0

from const_simu import constellation

service_delta_const = constellation.ServiceDeltaConstellation(
    num_orbits=num_orbits,
    num_sats_per_orbit=num_sats_per_orbit,
    radius=radius,
    inclination=inclination,
    phasediff=phasediff,
    fs=fs,
)

# from const_simu import network_no_MPC
from const_simu import network_final_ver

service_delta_nwk = network_final_ver.ServiceSatNetDelta(
    service_delta_const,
    N_max,
    a_max,
    task_workload,
    link_bw,
    delta_t,
    max_time_span,
    bw_min,
    bw_max,
)

# service_delta_nwk.show_nwk_state_figure()
# service_delta_nwk.show_earth_figure()

service_delta_nwk.full_simu()
service_delta_nwk.const_size_test()
