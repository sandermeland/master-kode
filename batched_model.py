
import gurobipy as gp
import pandas as pd
from code_map import final_markets, new_meters, utils, analysis, timeframes, data_handling
import numpy as np
import pickle
from datetime import datetime


def run_batched_optimization_model_from_scratch(timeframe : timeframes.TimeFrame):
    L, M, F, H, freq_data, power_meter_dict, consumption_data = utils.get_all_sets(timeframe)
    L_u, L_d, Fu_h_l, Fd_h_l, R_h_l, P_h_m, Vp_h_m, Vm_m, R_m = utils.get_parameters(L,M,H)

    markets_name_dict = {market.name: market for market in M}
    market_names = list(markets_name_dict.keys())
    price_list = [market.price_data for market in M]
    volume_list = [market.volume_data for market in M]

    print(f"Amount of markets : {len(M)}")
    print(f"Amount of meters : {len(L)}")
    print(f"Amount of meters with direction up or both : {len(L_u)}")
    print(f"Amount of meters with direction down or both : {len(L_d)}")
    print(f"Amount of hours : {len(H)}")

    total_up_flex = np.sum(Fu_h_l) # total available flex volume up
    total_down_flex = np.sum(Fd_h_l) # total available flex volume down
    total_response_time = np.sum(R_h_l) # total response time
    #total_flex = total_up_flex + total_down_flex
    average_response_time = total_response_time/ (len(H)*len(L))
    hourly_flex_up = total_up_flex/len(H)
    hourly_flex_down = total_down_flex/len(H)

    print(f"Total up flex volume: {total_up_flex} MW")
    print(f"Total down flex volume: {total_down_flex} MW")
    print(f"Average flex volume pr hour up: {hourly_flex_up} MWh")
    print(f"Average flex volume pr hour down: {hourly_flex_down} MWh")
    print(f"Average response time: {average_response_time} seconds")
    compatible_list = utils.get_compatibility_dict(L, M)

    aggregated_results  = utils.run_batched_optimization_model(L= L , M = M, H = H, F = F, freq_data=freq_data, P_h_m=P_h_m , Vp_h_m =Vp_h_m, Vm_m = Vm_m, R_m = R_m, R_h_l = R_h_l, Fu_h_l = Fu_h_l, Fd_h_l = Fd_h_l, compatible_list = compatible_list, log_filename="batched_week_v5.log", model_name= "weekly_model_batched")

    return aggregated_results


