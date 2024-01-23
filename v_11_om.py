import gurobipy as gp
import pandas as pd
from code_map import final_markets, new_meters, utils, analysis, timeframes, data_handling
import numpy as np
import pickle
from datetime import datetime
import math
import plotly.graph_objects as go


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


def get_market_count_dict(x : dict, w: dict, H : [pd.Timestamp], L : [new_meters.PowerMeter], M : [final_markets.ReserveMarket], Fu_h_l : np.array, Fd_h_l : np.array, Ia_hlm : np.array, Ir_hlm : np.array):
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
        capacity_incomes = []
        activation_incomes = []
        for market_name in market_count["Market"]:
            m = market_names.index(market_name)
            market = M[m]
            capacity_income = sum(x[h, l, m].X * Ir_hlm[h, l, m] for l in range(len(L)))
            activation_income = sum(x[h, l, m].X * Ia_hlm[h, l, m] * w[h,m].X for l in range(len(L)))

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
            capacity_incomes.append(capacity_income)
            activation_incomes.append(activation_income)
        market_count["Total Flex Volume [MW]"] = flex_volumes
        market_count["Total capacity incume [EUR]"] = capacity_incomes
        market_count["Total activation income [EUR]"] = activation_incomes
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
        mc_dict = get_market_count_dict(x = x, w =w, H= batch_H,L= L, M= M, Fu_h_l= batch_Fu_h_l,Fd_h_l= batch_Fd_h_l, Ia_hlm=batch_Ia_hlm, Ir_hlm= batch_Ir_hlm)
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
        market_count_dict = get_market_count_dict(x, w, H, L, M, Fu_h_l, Fd_h_l, Ia_hlm, Ir_hlm)
        return model, x, y, w , market_count_dict
    

def run_optimization_model_part_of_tf(batch_tf : timeframes.TimeFrame, L : [new_meters.PowerMeter], M : [final_markets.ReserveMarket], H : [pd.Timestamp], P_h_m : np.array,  Vp_h_m : np.array, Vm_m : list, R_m : list, R_h_l : np.array, Fu_h_l : np.array, Fd_h_l : np.array, compatible_list : dict, log_filename : str, model_name : str):
    """ Function to create and run an optimization model for bidding in the reserve markets for a given set of meters and markets. The bidding is for historical data.
        This is a new version of the original run_optimization_model() function which can be found in the utils.py file. 
        This version is made to be able to run the model for a part of the timeframe.

    Args:
        batch_tf (timeframes.TimeFrame): the timeframe for the model to be run. If the sets are made from a timeframe which is larger than this timeframe, the model should be able to run only for this timeframe.
        L (list(new_meters.PowerMeter]): set of all meters
        M (list(final_markets.ReserveMarket]): set of all markets
        H (list(pd.Timestamp]): set of all hours
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
        test_model (gp.Model): The model that was run
        x (dict): The decision variables x[h,l,m] which tells if asset l is connected to market m at hour h
        y (dict): The decision variables y[h,m] which tells if market m has a bid at hour h
        w (dict): The decision variables w[h,m] which tells if market m is activated at hour h
        d (dict): The decision variables d[h,l,m] which tells if asset l is compatible with market m at hour h
    """

    # Slice numpy arrays for the current batch
    batch_H = utils.get_timestamps(batch_tf)
    start_index = list(H).index(batch_H[0])
    end_index = list(H).index(batch_H[-1]) + 1
    batch_R_h_l = R_h_l[start_index:end_index, :]
    batch_Fu_h_l = Fu_h_l[start_index:end_index, :]
    batch_Fd_h_l = Fd_h_l[start_index:end_index, :]
    batch_Vp_h_m = Vp_h_m[start_index:end_index, :]
    batch_P_h_m = P_h_m[start_index:end_index, :]
    batch_F, batch_freq_data, _ = utils.get_frequency_sets(tf = batch_tf, H = batch_H, M = M)
    # the income
    market_name_dict = {m.name : m for m in M}
    batch_Ir_hlm, batch_Ia_hlm, batch_Va_hm = utils.get_income_dictionaries(H = batch_H, M = M, L = L, freq_data = batch_freq_data, Fu_h_l = batch_Fu_h_l, Fd_h_l = batch_Fd_h_l, P_h_m = batch_P_h_m, Vp_h_m = batch_Vp_h_m, F = batch_F, markets_dict = market_name_dict, timeframe = batch_tf)
    
    # Create a new model
    model = gp.Model(model_name)
    model.setParam('OutputFlag', 1)
    model.setParam('LogFile', log_filename)

    # Create decision variables
    x = {}
    d = {}
    y = {}
    w = {}
    for h in range(len(batch_H)):
        for l in range(len(L)):
            for m in range(len(M)):
                # asset i is connected to market j at hour h
                x[h, l, m] = model.addVar(lb = 0, ub = 1, vtype=gp.GRB.BINARY, name=f"x_{h}_{l}_{m}")

                d[h,l,m] = 1 if l in compatible_list[m] else 0 # compatible_list takes care of both the area constraint and the direction constraint
                
                # adding the constraint
                model.addConstr(x[h,l,m] <= d[h,l,m]) # if a load is not compatible with market m it cant be connected to it
        for m in range(len(M)):
            # market m has a bid at hour h
            y[h, m] = model.addVar(lb = 0, ub = 1, vtype=gp.GRB.BINARY, name=f"y_{h}_{m}")
            # market m is activated at hour h
            w[h, m] = model.addVar(lb = 0, ub = 1, vtype=gp.GRB.BINARY , name=f"w_{h}_{m}")
            
    # Set the objective to maximize the total income expression
    model.setObjective(sum(x[h,l,m] * (batch_Ir_hlm[h,l,m] + batch_Ia_hlm[h,l,m] * w[h,m]) for h in range(len(batch_H)) for l in range(len(L)) for m in range(len(M))), gp.GRB.MAXIMIZE) # can possibly remove the x on the activation income

    # Add constraints
    for h in range(len(batch_H)):
        for l in range(len(L)):
            # Each asset can only be connected to one market at a time
            model.addConstr(sum(x[h, l, m] for m in range(len(M))) <= 1, f"single_market_for_asset_at_hour_{h}_nr.{l}")
        
        for m, market in enumerate(M):
            up_val, down_val = batch_F[h,m]
            if up_val + down_val > 0:
                model.addConstr(w[h,m] <= y[h,m], f"market_{m}_can_not_be_activated_at_hour_{h}_if_it_is_not_active")
            else:
                model.addConstr(w[h,m] == 0, f"market_{m}_can_not_be_activated_at_hour_{h}_if_it_is_not_active")
            
            # Connect the binary variables by using big M
            model.addConstr(sum(x[h, l, m] for l in range(len(L))) <= len(L) * y[h, m], f"asset_connection_for_hour_{h}_market_{m}")
        
            # Max volume constraint
            
            if market.direction == "up":
                # capacity volume constraint
                model.addConstr(sum(x[h, l, m] * batch_Fu_h_l[h,l] for l in range(len(L))) <= batch_Vp_h_m[h,m]  * y[h,m], f"max_volume_for_hour_{h}_market_{m}")
                # activation volume constraint
                model.addConstr(sum(x[h, l, m] * batch_Fu_h_l[h,l] for l in range(len(L))) * w[h,m] <= batch_Va_hm[h,m], f"max_volume_for_activation_in-_market_{m}_at_hour_{h}")
                # min volume capacity constraint
                model.addConstr(sum(x[h, l, m] * batch_Fu_h_l[h,l] for l in range(len(L))) >= Vm_m[m] * y[h, m], f"min_volume_for_hour_{h}_market_{m}") 

            elif market.direction == "down":
                # max capacity volume constraint
                model.addConstr(sum(x[h, l, m] * batch_Fd_h_l[h,l] for l in range(len(L))) <= batch_Vp_h_m[h,m]  * y[h,m], f"max_volume_for_hour_{h}_market_{m}")
                # max activation volume constraint
                model.addConstr(sum(x[h, l, m] * batch_Fd_h_l[h,l] for l in range(len(L))) * w[h,m] <= batch_Va_hm[h,m], f"max_volume_for_activation_in_market_{m}_at_hour_{h}")
                # min volume capacity constraint
                model.addConstr(sum(x[h, l, m] * batch_Fd_h_l[h,l] for l in range(len(L))) >= Vm_m[m] * y[h, m], f"min_volume_for_hour_{h}_market_{m}") 

            else: # market.direction == "both" => In FCR-N you must be able to activate in both directions
                # max capacity volume constraint
                model.addConstr(sum(x[h, l, m] * batch_Fu_h_l[h,l] for l in range(len(L))) <= batch_Vp_h_m[h,m]  * y[h,m], f"max_volume_for_hour_{h}_market_{m}")
                # max activation volume constraint
                model.addConstr(sum(x[h, l, m] * batch_Fu_h_l[h,l] for l in range(len(L))) * w[h,m] <= batch_Va_hm[h,m] , f"max_volume_for_activation_in_market_{m}_at_hour_{h}")
                # min capacity volume constraint
                model.addConstr(sum(x[h, l, m] * batch_Fu_h_l[h,l] for l in range(len(L))) >= Vm_m[m] * y[h, m], f"min_volume_for_hour_{h}_market_{m}") 
                
                # max capacity volume constraint
                model.addConstr(sum(x[h, l, m] * batch_Fd_h_l[h,l] for l in range(len(L))) <= batch_Vp_h_m[h,m] * y[h,m], f"max_volume_for_hour_{h}_market_{m}")
                # max activation volume constraint
                model.addConstr(sum(x[h, l, m] * batch_Fd_h_l[h,l] for l in range(len(L))) * w[h,m] <= batch_Va_hm[h,m], f"max_volume_for_activation_in_market_{m}_at_hour_{h}")
                # min capacity volume constraint
                model.addConstr(sum(x[h, l, m] * batch_Fd_h_l[h,l] for l in range(len(L))) >= Vm_m[m] * y[h, m], f"min_volume_for_hour_{h}_market_{m}") 
            
                # add a constraint where the reserved volume in the FCR_N markets has to be the same for each direction
                # I cant find a way to do this in a mathematical model, i think it should be held out of the equation for this part and rather use it in the dynamic model
                #model.addConstr(sum(x[h, l, m] * Fu_h_l[h,l] for l in range(len(L))) == sum(x[h, l, m] * Fd_h_l[h,l] for l in range(len(L)))) # this can be hard as they have to be exactly equal
               

            # The response times for loads l connected to market m cannot exceed the max response time for m
            for l in range(len(L)):
                model.addConstr(x[h,l,m] * batch_R_h_l[h,l] <= R_m[m] * y[h,m], f"response_time_for_hour_{h}_market_{m}")
                
    model.optimize(utils.callback_factory(log_filename))

    if model.status == gp.GRB.Status.INFEASIBLE:
        model.computeIIS()
        
    market_count_dict = get_market_count_dict(x = x, w = w, H=  batch_H, L=  L,M = M, Fu_h_l= batch_Fu_h_l, Fd_h_l=batch_Fd_h_l, Ia_hlm= batch_Ia_hlm, Ir_hlm= batch_Ir_hlm) # there might be an issue here as it uses the large timeframe ???

    return model, x, y, w, market_count_dict


def run_and_compare_solutions_larger_batched(tf_big, tf_small, lf_ord, lf_batched, mn_ord, mn_batched):
    """ This function is made to compare the results from the batched model and the ordinary model when the batched model is run for a larger timeframe than the ordinary model.
        The function has been tested and it works as intended for big tf = 3 days and small tf = 1 day.

    Args:
        tf_big (timeframes.Timeframe): the timeframe for the batched model
        tf_small (timeframes.TimeFrame): the timeframe fr the ordinary model
        lf_ord (str): the name of the log file for the ordinary model
        lf_batched (str): the name of the log file for the batched model
        mn_ord (str): the name of the model for the ordinary model
        mn_batched (str): the name of the model for the batched model

    Returns:
        mod (gurobi model) : the model that is returned from the ordinary model
        x (dict) : the x-values for the ordinary model
        y (dict) : the y-values for the ordinary model
        w (dict) : the w-values for the ordinary model
        market_count_dict (dict) : the results from the ordinary model
        agg_res (dict) : the results from the batched model
        results_dict (list) : list of the results from the batched model, where the index is the batch number / day number
    """
    L, M, F, H, _, Fu_h_l, Fd_h_l, R_h_l, P_h_m, Vp_h_m, Vm_m, R_m, _, _, Ir_hlm, Ia_hlm, Va_hm, compatible_list = get_collections(tf_big)
    agg_res, results_dict = run_batched_optimization_model(L = L, M = M, H = H, Fu_h_l = Fu_h_l, Fd_h_l = Fd_h_l, R_h_l = R_h_l, P_h_m = P_h_m, Vp_h_m = Vp_h_m, Vm_m = Vm_m, R_m = R_m, compatible_list = compatible_list, model_name = mn_batched, log_filename = lf_batched)
    mod, x , y, w , market_count_dict= run_optimization_model_part_of_tf(batch_tf = tf_small, L = L, M = M, H = H, P_h_m = P_h_m,  Vp_h_m = Vp_h_m, Vm_m = Vm_m, R_m =R_m, R_h_l = R_h_l, Fu_h_l = Fu_h_l, Fd_h_l =Fd_h_l, compatible_list = compatible_list, log_filename =lf_ord, model_name = mn_ord)
    return mod, x, y, w, market_count_dict, agg_res, results_dict


timeframe = timeframes.TimeFrame(year = 2023, start_month = 6, end_month = 6, start_day = 20, end_day = 22, start_hour = 0, end_hour = 23)
sub_timeframe = timeframes.TimeFrame(year = 2023, start_month = 6, end_month = 6, start_day = 21, end_day = 21, start_hour = 0, end_hour = 23)


mod, x, y, w, market_count_dict, aggregated_results, results_dict = run_and_compare_solutions_larger_batched(tf_big= timeframe, tf_small = sub_timeframe, lf_ord = "ordinary_one_day.log", lf_batched = "batched_three_days.log", mn_ord = "ordinary_model_one_day", mn_batched = "batched_model_three_days")


with open("results_three_and_one_days_model.txt", "w") as f:
    f.write(f"Results from ordinary model and batched model where batched models are run for three days and ordinary model is run for one day. \n")
    equal_count = 0
    total_count = 0
    for key in market_count_dict.keys():
        for i in range(len(results_dict)):
            if key in results_dict[i].keys():
                f.write(f"Hour {key} \n")
                f.write(f"Result from batched model \n {results_dict[i][key]} \n")
                f.write(f"Result from ordinary model \n {market_count_dict[key]} \n")
                f.write(f"Equal :  {results_dict[i][key].equals(market_count_dict[key])} \n")
                total_count += 1
                if results_dict[i][key].equals(market_count_dict[key]):
                    equal_count += 1
            else:
                continue
    f.write(f"Amount of equal results : {equal_count} / {total_count} \n")


"""test_hour = pd.Timestamp(year = 2023, month = 6, day = 21, hour = 14, tz= "Europe/Oslo")
len(results_dict[1][test_hour])
market_count_dict[test_hour]["Total Flex Volume [MW]"]

combined_df_batch = pd.concat([pd.concat(results_dict[i], ignore_index=True) for i in range(len(results_dict))], ignore_index=True)

sum(combined_df_batch["Asset Count"].loc[combined_df_batch["Hour"] == test_hour])"""

# Asset Count  Total Flex Volume [MW]  Total capacity incume [EUR/MW]  Total activation income [EUR/MW]
def plot_results(results_dict : list, market_count_dict : dict, aggregated : bool = False, both_results : bool = False):
    
    #concatenate all the results from the batched model where the results are stored as dataframes in a dict for each hour inside a list for each episode
    combined_df_batch = pd.concat([pd.concat(results_dict[i], ignore_index=True) for i in range(len(results_dict))], ignore_index=True)
    #concatenate all the results from the ordinary model where the results are stored as dataframes in a dict for each hour
    combined_df_org = pd.concat(market_count_dict, ignore_index=True)
    
    for column in ["Asset Count",  "Total Flex Volume [MW]",  "Total capacity incume [EUR/MW]",  "Total activation income [EUR/MW]"]:
        fig = go.Figure()
        if aggregated:
            fig.add_trace(go.Scatter(x = combined_df_batch["Hour"], y = [sum(combined_df_batch[column].loc[combined_df_batch["Hour"] == hour]) for hour in combined_df_batch["Hour"]], mode = "lines", name = f"{column} in batched model"))
            if both_results:
                fig.add_trace(go.Scatter(x = combined_df_org["Hour"], y = [sum(combined_df_org[column].loc[combined_df_org["Hour"] == hour]) for hour in combined_df_org["Hour"]], mode = "lines", name = f"{column} in ordinary model"))
        else:
            for market in combined_df_batch["Market"].unique():
                market_data = combined_df_batch.loc[combined_df_batch["Market"] == market]
                fig.add_trace(go.Scatter(x = market_data["Hour"], y = market_data[column], mode = "markers", name = f"{market} in batched model"))
                if both_results:
                    fig.add_trace(go.Scatter(x = combined_df_org["Hour"], y = combined_df_org.loc[combined_df_org["Market"] == market, column], mode = "markers", name = f"{column} for {market} ordinary model"))
        fig.update_xaxes(title_text="Hours")
        fig.update_yaxes(title_text= column)
        fig.update_layout(title_text=f"{column} by Market and Hour from the optimization model")
        fig.show()
    return None

plot_results(results_dict = results_dict, market_count_dict = market_count_dict, aggregated= True)






# These methods are when the batched version and the orig version have the same timeframes
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
        
