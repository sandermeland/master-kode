import gurobipy as gp
import pandas as pd
from code_map import final_markets, new_meters, utils, analysis, timeframes, data_handling
import numpy as np
import pickle
from datetime import datetime
import math

def get_collections(timeframe : timeframes.TimeFrame):
    L, M, H = utils.get_all_sets(timeframe)
    F, freq_data, _ = utils.get_frequency_sets(timeframe, H, M)
    L_u, L_d, Fu_h_l, Fd_h_l, R_h_l, P_h_m, Vp_h_m, Vm_m, R_m = utils.get_parameters(L,M,H)
    markets_name_dict = {market.name: market for market in M}
    print(f"Amount of markets : {len(M)}")
    print(f"Amount of meters : {len(L)}")
    print(f"Amount of meters with direction up or both : {len(L_u)}")
    print(f"Amount of meters with direction down or both : {len(L_d)}")
    print(f"Amount of hours : {len(H)}")
    dominant_directions = [utils.get_dominant_direction(freq_data, hour) for hour in H]
    Ir_hlm, Ia_hlm, Va_hm = utils.get_income_dictionaries(H = H, M = M, L = L, freq_data= freq_data, Fu_h_l = Fu_h_l, Fd_h_l = Fd_h_l, P_h_m = P_h_m, Vp_h_m = Vp_h_m, F = F, markets_dict= markets_name_dict, timeframe = timeframe)
    compatible_list = utils.get_compatibility_dict(L, M)
    return L, M, F, H, freq_data, Fu_h_l, Fd_h_l, R_h_l, P_h_m, Vp_h_m, Vm_m, R_m, markets_name_dict, dominant_directions, Ir_hlm, Ia_hlm, Va_hm, compatible_list


def get_market_count_dict(x : dict, H : [pd.Timestamp], L : [new_meters.PowerMeter], M : [final_markets.ReserveMarket], Fu_h_l : np.array, Fd_h_l : np.array):
    """ function to get a dictionary of the results of the optimization problem.
        The solution is represented as a dataframe for each hour which tells how many assets and how much flex volume is connected to each market for each hour.

    Args:
        x (dict): dictionary of the binary variable which tells if an asset is connected to a market
        L (list(PowerMeter)): list of powermeter objects with the data for each meter within the timeframe
        M (list(ReserveMarket)): list of reservemarket objects with the data for each market within the timeframe
        H (list(pd.TimeStamp)): list of hourly timestamps within the timeframe
        dominant_directions (list(str)): list of the dominant direction for each hour

    Returns:
        dict: the solution of the optimization problem
    """
    data = []

    for h, hour in enumerate(H):
        for l, load in enumerate(L):
            for m, market in enumerate(M):
                if x[h, l, m].X > 0.5:
                    # Calculate flex volume for this asset, market, and hour
        
                    data.append([hour, load.meter_id, market.name])

    df = pd.DataFrame(data, columns=["Hour", "Asset Meter ID", "Market"])
    market_names = [m.name for m in M]
    market_count_dict = {}
    for h, hour in enumerate(H):
        hour_df = df.loc[(df["Hour"] == hour)]
        # Aggregate data by market and hour, counting assets and summing flex volumes
        market_count = hour_df.groupby(["Market", "Hour"]).agg({"Asset Meter ID": "count"}).reset_index().rename(columns={"Asset Meter ID": "Asset Count"})
        flex_volumes = []
        for market_name in market_count["Market"]:
            m = market_names.index(market_name)
            market = M[m]
            if market.direction == "up":
                total_flex_volume = sum(x[h, l, m].X *Fu_h_l[h,l] for l, load in enumerate(L) if load.direction != "down")
            elif market.direction == "down":
                total_flex_volume = sum(x[h, l, m].X * Fd_h_l[h,l] for l, load in enumerate(L) if load.direction != "up")
            else:
                total_flex_volume = 0
                for l, load in enumerate(L):
                    if load.direction == "up":
                        total_flex_volume += x[h, l, m].X * Fu_h_l[h,l]
                    elif load.direction == "down":
                        total_flex_volume += x[h, l, m].X * Fd_h_l[h,l]
                    else:
                        total_flex_volume += min(x[h, l, m].X * Fu_h_l[h,l], x[h, l, m].X * Fd_h_l[h,l])
                
            flex_volumes.append(total_flex_volume)
        market_count["Total Flex Volume"] = flex_volumes
        market_count_dict[hour] = market_count
    return market_count_dict

def run_batched_optimization_model(L : [new_meters.PowerMeter], M : [final_markets.ReserveMarket], H : [pd.Timestamp], P_h_m : np.array, Vp_h_m : np.array, Vm_m : list, R_m : list, R_h_l : np.array, Fu_h_l : np.array, Fd_h_l : np.array, compatible_list : dict, log_filename : str, model_name : str):
    """ Function to create and run an optimization model for bidding in the reserve markets for a given set of meters and markets. 
    The bidding is for historical data. The model is run in batches to avoid memory issues. Therefore the H list is splitted in to batches of 24 hours. So the model is run for each batch.
    The parameters that has values for h are splitted in to batches of the same length as the H batch to make sure that the values are correct for each batch.

    Args:
        L (list(new_meters.PowerMeter]): set of all meters
        M (list(final_markets.ReserveMarket]): set of all markets
        H (list(pd.Timestamp]): set of all hours
        F (dict): Dictionary to find the activation percentages for each market and hour
        freq_data (pd.DataFrame): dataframe with the frequency data
        P_h_m (np.array): The price for each hour and market
        Vp_h_m (np.array): The volume for each hour and market
        Vm_m (list): Minimum volume for each market
        R_m (list): Response time for each market
        R_h_l (np.array): Response time for each load each hour
        Fu_h_l (np.array): Up flex volume for each load that are compatible with up markets for each hour 
        Fd_h_l (np.array): Down flex volume for each load that are compatible with down markets for each hour
        compatible_list (dict): dict of compatible markets for each asset
        log_filename (str): name of the logfile
        model_name (str): name of the model

    Returns:
        aggregated_results (dict): The results from the batched optimization model which includes the model, the decision variables and the values of the decision variables
            model (gp.Model): The model that was run
            x (dict): The decision variables x[h,l,m] which tells if asset l is connected to market m at hour h
            y (dict): The decision variables y[h,m] which tells if market m has a bid at hour h
            w (dict): The decision variables w[h,m] which tells if market m is activated at hour h
            d (dict): The decision variables d[h,l,m] which tells if asset l is compatible with market m at hour h
    """
    batch_size = 24  # For example, batching by 24 hours
    num_batches = math.ceil(len(H) / batch_size)
    aggregated_results = {
        'models': [],
        'x_values': [],
        'y_values': [],
        'w_values': [],
        'd_values': []
    }
    market_name_dict = {m.name : m for m in M}
    result_dicts = []
    for b in range(num_batches):
        # Determine the subset of hours for this batch
        start_index = b * batch_size
        end_index = min((b + 1) * batch_size, len(H))
        batch_H = H[start_index:end_index]

        # Slice numpy arrays for the current batch
        batch_R_h_l = R_h_l[start_index:end_index, :]
        batch_Fu_h_l = Fu_h_l[start_index:end_index, :]
        batch_Fd_h_l = Fd_h_l[start_index:end_index, :]
        batch_Vp_h_m = Vp_h_m[start_index:end_index, :]
        batch_P_h_m = P_h_m[start_index:end_index, :]
        batch_tf = timeframes.TimeFrame(year = 2023, start_month = 6, end_month = 6, start_day = batch_H[0].day, end_day = batch_H[0].day, start_hour = 0, end_hour = 23)
        batch_F, batch_freq_data, _ = utils.get_frequency_sets(tf = batch_tf, H = batch_H, M = M)
        # the income
        batch_Ir_hlm, batch_Ia_hlm, batch_Va_hm = utils.get_income_dictionaries(H = batch_H, M = M, L = L, freq_data = batch_freq_data, Fu_h_l = batch_Fu_h_l, Fd_h_l = batch_Fd_h_l, P_h_m = batch_P_h_m, Vp_h_m = batch_Vp_h_m, F = batch_F, markets_dict = market_name_dict, timeframe = batch_tf)
       
        # Run the optimization model for this batch
        model, x, y, w, _ = utils.run_optimization_model(L= L, M= M, H = batch_H,F= batch_F, Ir_hlm= batch_Ir_hlm, Ia_hlm= batch_Ia_hlm, Va_hm= batch_Va_hm, Vp_h_m= batch_Vp_h_m, Vm_m=Vm_m, R_m=R_m, R_h_l=batch_R_h_l, Fu_h_l=batch_Fu_h_l, Fd_h_l=batch_Fd_h_l, compatible_list=compatible_list, log_filename=log_filename, model_name=f"{model_name}_batch_{b}")
        # Store results
        aggregated_results['models'].append(model)
        aggregated_results['x_values'].append(x)
        aggregated_results['y_values'].append(y)
        aggregated_results['w_values'].append(w)
        #aggregated_results['d_values'].append(d)
        #utils.test_solution_validity(x, y, w, batch_Va_hm, L, M, batch_H, F)
        mc_dict = get_market_count_dict(x, batch_H, L, M, batch_Fu_h_l, batch_Fd_h_l)
        result_dicts.append(mc_dict)

    # Process aggregated_results as needed
    return aggregated_results, result_dicts


def run_one_optimization_model(timeframe : timeframes.TimeFrame, log_filename : str, model_name : str, batched : bool = False):
    L, M, F, H, freq_data, Fu_h_l, Fd_h_l, R_h_l, P_h_m, Vp_h_m, Vm_m, R_m, markets_name_dict, dominant_directions, Ir_hlm, Ia_hlm, Va_hm, compatible_list = get_collections(timeframe)
    if batched:
        agg_res, results_dict = run_batched_optimization_model(L = L, M = M, H = H, F = F, freq_data= freq_data, Fu_h_l = Fu_h_l, Fd_h_l = Fd_h_l, R_h_l = R_h_l, P_h_m = P_h_m, Vp_h_m = Vp_h_m, Vm_m = Vm_m, R_m = R_m, compatible_list = compatible_list, model_name = model_name, log_filename = log_filename)
        return agg_res, results_dict
    else:
        model, x , y, w ,_ = utils.run_optimization_model(L = L, M = M, H = H, F = F, Fu_h_l = Fu_h_l, Fd_h_l = Fd_h_l, R_h_l = R_h_l,  Vp_h_m = Vp_h_m, Vm_m = Vm_m, R_m = R_m, Ir_hlm = Ir_hlm, Ia_hlm = Ia_hlm, Va_hm = Va_hm, compatible_list = compatible_list, model_name = model_name, log_filename = log_filename)
        market_count_dict = get_market_count_dict(x, H, L, M, Fu_h_l, Fd_h_l)
        return model, x , y, w , market_count_dict
    

def run_and_compare_solutions(tf, lf_ord, lf_batched, mn_ord, mn_batched):
    L, M, F, H, _, Fu_h_l, Fd_h_l, R_h_l, P_h_m, Vp_h_m, Vm_m, R_m, _, _, Ir_hlm, Ia_hlm, Va_hm, compatible_list = get_collections(tf)
    agg_res, results_dict = run_batched_optimization_model(L = L, M = M, H = H, Fu_h_l = Fu_h_l, Fd_h_l = Fd_h_l, R_h_l = R_h_l, P_h_m = P_h_m, Vp_h_m = Vp_h_m, Vm_m = Vm_m, R_m = R_m, compatible_list = compatible_list, model_name = mn_batched, log_filename = lf_batched)
    mod, x , y, w ,_ = utils.run_optimization_model(L = L, M = M, H = H, F = F, Fu_h_l = Fu_h_l, Fd_h_l = Fd_h_l, R_h_l = R_h_l,  Vp_h_m = Vp_h_m, Vm_m = Vm_m, R_m = R_m, Ir_hlm = Ir_hlm, Ia_hlm = Ia_hlm, Va_hm = Va_hm, compatible_list = compatible_list, model_name = mn_ord, log_filename = lf_ord)
    market_count_dict = get_market_count_dict(x, H, L, M, Fu_h_l, Fd_h_l)
    return mod, x, y, w, market_count_dict, agg_res, results_dict


mod, x, y, w, market_count_dict, aggregated_results, results_dict = run_and_compare_solutions(tf= timeframes.two_days, lf_ord = "ordinary_two_days.log", lf_batched = "batched_two_days.log", mn_ord = "ordinary_model_two_days", mn_batched = "batched_model_two_days")



# print the results in a log file
with open("results_two_days_model_two_versions.txt", "w") as f:
    f.write(f"Results from ordinary model and batched model where both models are run for two days where the collections was loaded one time. \n")
    equal_count = 0
    total_count = 0
    for i in range(len(results_dict)):
        f.write(f"Batch {i} \n")
        for key in results_dict[i].keys():
            f.write(f"Hour {key} \n")
            f.write(f"Result from batched model \n {results_dict[i][key]} \n")
            f.write(f"Result from ordinary model \n {market_count_dict[key]} \n")
            f.write(f"Equal :  {results_dict[i][key].equals(market_count_dict[key])} \n")
            total_count += 1
            if results_dict[i][key].equals(market_count_dict[key]):
                equal_count += 1
    f.write(f"Amount of equal results : {equal_count} / {total_count} \n")
        


