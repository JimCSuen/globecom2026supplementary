class result_record_class:
    def __init__(self):
        self.record_list = []

    def add_record(
        self,
        offload_cost,
        link_LB,
        sat_pLB,
        max_link_load_ratio,
        max_sat_pload,
        low_task_size,
        high_task_size,
        mean_task_num,
        V_param,
        N_const,
    ):
        self.record_list.append(
            {
                "offload_cost": offload_cost,  # * only for the lyap-MPC method
                "link_load_ratio": link_LB,
                "sat_pload_ratio": sat_pLB,
                "max_link_load_ratio": max_link_load_ratio,
                "max_sat_pload": max_sat_pload,
                "low_task_size": low_task_size,
                "high_task_size": high_task_size,
                "mean_task_num": mean_task_num,
                "V_param": V_param,  # * only for the lyap-MPC method
                "const_size": N_const,
            }
        )
