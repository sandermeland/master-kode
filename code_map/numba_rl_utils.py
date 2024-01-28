import pandas as pd
from code_map import final_markets, new_meters, utils, weather, timeframes
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import random
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numba
import statistics


@numba.jit(nopython=True)
def random_arg_max(possible_actions : np.array(float)):
    imax = 0
    xmax = possible_actions[imax]  # Current maximum
    nmax = 1  # Number of maximum values at the moment
    for i in range(1, len(possible_actions)):
        if possible_actions[i] == xmax:
            nmax += 1
            if nmax * random.random() < 1.0:
                imax = i
        elif possible_actions[i] > xmax:
            nmax = 1  # Reset count since a new maximum is found
            imax = i
            xmax = possible_actions[i]  # Update the new maximum
    return imax

@numba.jit(nopython=True)
def greedy_action(possible_actions : np.array(float), epsilon : float ):
    """returns the index of the greedy action

    Args:
        possible_actions (lsit): list of the possible actions
        epsilon (float): float number between 0 and 1, often close to 0

    Returns:
        int: index of the greedy action
    """
    if np.random.rand() <= (1- epsilon): # pick greedy
        return random_arg_max(possible_actions)
    else:
        return np.random.randint(0, len(possible_actions)-1) # random
    
def get_market_and_asset_values(H, L, M):
    up_volumes_hl = np.zeros((len(H), len(L)))
    down_volumes_hl = np.zeros((len(H), len(L)))
    market_prices_hm = np.zeros((len(H), len(M)))
    market_volumes_hm = np.zeros((len(H), len(M)))
    market_directions_m = np.zeros((len(M)))
    asset_directions_l = np.zeros((len(L)))

    for l, asset in enumerate(L):
        asset_directions_l[l] = 1 if asset.direction == "up" else -1 if asset.direction == "down" else 0
        for h, hour in enumerate(H):
            up_volumes_hl[h, l] = asset.up_flex_volume["value"].loc[asset.up_flex_volume["Time(Local)"] == hour].values[0] if asset.direction != "down" else 0
            down_volumes_hl[h,l] = asset.down_flex_volume["value"].loc[asset.down_flex_volume["Time(Local)"] == hour].values[0] if asset.direction != "up" else 0
    for m, market in enumerate(M):
        market_directions_m[m] = 1 if market.direction == "up" else -1 if market.direction == "down" else 0
        for h, hour in enumerate(H):
            market_prices_hm[h, m] = market.price_data.loc[market.price_data["Time(Local)"] == hour].values[0][1]
            market_volumes_hm[h, m] = market.volume_data.loc[market.volume_data["Time(Local)"] == hour].values[0][1]
    
    return up_volumes_hl, down_volumes_hl, market_prices_hm, market_volumes_hm, asset_directions_l, market_directions_m

def get_expected_prices_and_volumes_dict(n_hours: int, market_volumes : np.array(float), market_prices : np.array(float), market_directions : np.array(int)):
    """ function to calculate the expected prices for each direction and area in the bid_timeframe

    Args:
        bid_timeframe (_type_): _description_
        markets (_type_): _description_

    Returns:
        _type_: _description_
    """
    expected_prices = np.zeros((3, n_hours)) # only NO5 is taken in to consideration
    expected_volumes = np.zeros((3, n_hours))
    for d in [-1, 0, 1]:
        #for area in ["NO1", "NO2", "NO3", "NO4", "NO5"]:
        for h in range(n_hours):
            expected_prices[(d, h)] = np.mean([market_prices[h, m] for m in range(len(market_directions)) if market_directions[m] == d])
            expected_volumes[(d, h)] = np.mean([market_volumes[h, m] for m in range(len(market_directions)) if market_directions[m] == d])
    return expected_prices, expected_volumes


def normalize_array_vals(array, norm_method :str = "min_max" ):
    """Function to normalize the values of a dictionary

    Args:
        dict (dict): [description]
        norm_method (str): Normalization method; can choose between min-max normalization or z-score normalization. Defaults to "min_max".

    Returns:
        dict: the same dictionary as input, but with normalized values
    """
    if norm_method == "min_max":
        # check how mamy dimensions the array has
        if len(array.shape) >= 1:
            min_vals = array.min(axis=1, keepdims=True)
            max_vals = array.max(axis=1, keepdims=True)
            normalized_array = (array - min_vals) / (max_vals - min_vals)   
        else:
            # if the array has only one dimension, we can simply normalize the values
            min_value = np.min(array)
            max_value = np.max(array)
            # normalize the values in the array
            normalized_array = (array - min_value) / (max_value - min_value)
    else:
        if len(array.shape) >= 1:
            mean_vals = array.mean(axis=1, keepdims=True)
            std_vals = array.std(axis=1, keepdims=True)
            normalized_array = (array - mean_vals) / std_vals
        else:
            mean = statistics.mean(array)
            std = statistics.stdev(array)
            # Normalize and update the dictionary
            normalized_array = (array - mean) / std

    return normalized_array

# the markets: ['FCR_D_D_1_NO5','FCR_D_D_2_NO5','FCR_N_D_1_NO5','FCR_N_D_2_NO5','aFRR up_NO5','aFRR down_NO5']


@numba.jit(nopython=True)
def get_possible_dates(hour_index : int):
    """ Function to get the possible dates for placing a bid given the current date

    Args:
        date_index (int): the index of the current hour in H

    Returns:
        (np.array(int), np.array(int)): array of the possible hour indeces for placing a bid in the next 24 hours and an array of the possible market indeces
    """
    if hour_index in [17 + 24*i for i in range(7)]: # FCR D-2
        return (np.array([i for i in range(hour_index+7, hour_index+24+7)]), np.array([1,3]))
    elif hour_index in [7 + 24*i for i in range(7)]: # aFRR
        return (np.array([i for i in range(hour_index+17, hour_index+24+17)]), np.array([4,5]))
    elif hour_index in [18 + 24*i for i in range(7)]: # FCR D-1
        return (np.array([i for i in range(hour_index+6, hour_index+24+6)]), np.array([0,2]))
    else:
        return ([], [])
    


def get_compatibility_array(L : [new_meters.PowerMeter], M : [final_markets.ReserveMarket], index = True, meter_id = False, array = False):
    """ function to get a dict of compatible markets for each asset. The dict is either a dict of lists of indexes or a dict of lists of objects.
        The function checks the response time, the area and the direction of the asset and the market to see if they are compatible

    Args:
        H (list(pd.TimeStamp)): list of hourly timestamps within the timeframe
        L (list(PowerMeter)): list of powermeter objects with the data for each meter within the timeframe
        M (list(ReserveMarket)): list of reservemarket objects with the data for each market within the timeframe

    Returns:
        dict: list of compatible loads for each market
    """
    compatible_array = np.zeros((len(L), len(M)))
    for l, asset in enumerate(L):
        for m, market in enumerate(M):
            if asset.response_time <= market.response_time and asset.area == market.area and asset.direction in market.direction:
                compatible_array[l, m] = 1
            else:
                compatible_array[l, m] = -1
    
    return compatible_array
#up_volumes_hl, down_volumes_hl, market_prices_hm, market_volumes_hm, asset_directions_l, market_directions_m

@numba.jit(nopython=True)
def get_n_portfolios_for_market(available_asset_indeces : np.array(int), market_index : int, hour_index : int, compatible_array : np.array(int), up_volumes_hl : np.array(int), down_volumes_hl: np.array(int), market_directions_m: np.array(int), top_n=100, min_n = 20, iterations=100, sorting : str = "max"):
    """function to find a diverse and large subset of feasible combinations without the computational overhead of checking all possible combinations.
        DISCLAIMER : This function doesnt take procured volume in the given market in to account.

    Args:
        possible_assets ([new_meters.PowerMeter]): list of possible assets for the given market and given hour
        market (final_markets.ReserveMarket): the market for which the portfolio is to be found
        hour (pd.Timestamp): the hour for which the portfolio is to be found
        compatible_dict (dict): dictionary of compatible assets for each market
        top_n (int, optional): controls the number of top assets to consider. Defaults to 100.
        iterations (int, optional):  determines how many different combinations to try and generate. Defaults to 100.

    Returns:
        feasible_combinations: dict of top_n feasible combinations of assets that can be bid to the given market in the given hour where the values holds the portfolio's aggregated volume
    """
    
    #feasible_assets = possible_asset_ids[compatible_array[:, market_index] == 1] # exclude assets that are not compatible with the given market
    # Fetch volumes for each asset at the given hour

    if market_directions_m[market_index] == 0:
        asset_volumes = np.array([(l, up_volumes_hl[hour_index, l], down_volumes_hl[hour_index, l]) for l in available_asset_indeces if compatible_array[l, market_index] == 1])
    elif market_directions_m[market_index] == 1:
        asset_volumes = np.array([(l, up_volumes_hl[hour_index, l]) for l in available_asset_indeces if compatible_array[l, market_index] == 1])
    else:
        asset_volumes = np.array([(l, down_volumes_hl[hour_index, l]) for l in available_asset_indeces if compatible_array[l, market_index] == 1])
       

    # Sort assets by volume in descending order
    sums  = np.zeros((len(asset_volumes)))
    if sorting == "random":
        np.random.shuffle(asset_volumes)
    else:
        if market_directions_m[market_index] == 0: # both
            for i in range(len(asset_volumes)):
                sums[i] = asset_volumes[i][1] + asset_volumes[i][2] # må kanskje bruke [i][1]
            # Get indices that would sort the array based on the sums
            sorted_indices = np.argsort(sums) if sorting == "max" else np.argsort(sums)[::-1]
            asset_volumes = asset_volumes[sorted_indices]
        else:
            for i in range(len(asset_volumes)):
                sums[i] = asset_volumes[i][1]
            sorted_indices = np.argsort(sums) if sorting == "max" else np.argsort(sums)[:-1]
            asset_volumes = asset_volumes[sorted_indices]
        
        
    # Select top N assets
    top_assets = asset_volumes[:top_n]
    #top_assets = asset_volumes

    # Check if total volume of top assets is greater than minimum volume
    if market_directions_m[market_index] == 0: # both
        total_up_volume = np.sum([asset[1] for asset in top_assets])
        total_down_volume = np.sum([asset[2] for asset in top_assets])
        if total_down_volume < 1 or total_up_volume < 1:
            #print("total volume is less than minimum volume")
            return [], 0
    else:
        total_volume = np.sum([asset[1] for asset in top_assets])
        if total_volume < 1:
            #print("total volume is less than minimum volume")
            return [], 0
    
    #feasible_combinations = set()
    for i in range(iterations):
        # Randomly sample a smaller subset from top assets
        num_assets_to_sample = np.min(len(top_assets), np.random.randint(min_n, top_n)) # not sure if this is necessary

        sampled_assets = random.sample(top_assets, k=num_assets_to_sample) 
        #sampled_assets = random.sample(top_assets, k=len(top_assets))
        #print(len(sampled_assets))

        # Greedy addition to meet minimum volume
        if market_directions_m[market_index] == 0: # both
            combination, total_up_volume , total_down_volume = [], 0, 0
            for asset, up_vol, down_vol in sampled_assets:
                combination.append(asset)
                total_up_volume += up_vol
                total_down_volume += down_vol
                if total_down_volume >= 1 and total_up_volume >= 1:
                    #feasible_combinations.add((tuple(combination), total_volume))
                    return np.array(combination), min(total_up_volume, total_down_volume)

        else:
            combination, total_volume = [], 0
            for asset, volume in sampled_assets:
                combination.append(asset)
                total_volume += volume
                if total_volume >= 1:
                    return np.array(combination), min(total_up_volume, total_down_volume)
    return [], 0

    



@numba.jit(nopython=True)
def get_max_portfolio_for_market(available_asset_indeces : np.array(int), market_index : int, hour_index : int, compatible_array : np.array(int), up_volumes_hl : np.array(int), down_volumes_hl: np.array(int), market_directions_m: np.array(int)):
    """ Heuristic to bid all feasible assets in to the market in given hour. Equivalent to greedy approach
        DISCLAIMER : This function doesnt take procured volume in the given market in to account.
    Args:
        possible_assets ([new_meters.PowerMeter]): list of possible assets for the given market and given hour
        market (final_markets.ReserveMarket): the market for which the portfolio is to be found
        hour (pd.Timestamp): the hour for which the portfolio is to be found
        compatible_dict (dict): dictionary of compatible assets for each market

    Returns:
        feasible_assets ([new_meters.PowerMeter]): list of the assets that will be bid in to the given market in the given hour
        total_volume (float): the total volume of the feasible assets for the given hour
    """
    if market_directions_m[market_index] == 0:
        asset_volumes = np.array([(l, up_volumes_hl[hour_index, l], down_volumes_hl[hour_index, l]) for l in available_asset_indeces if compatible_array[l, market_index] == 1])
    elif market_directions_m[market_index] == 1:
        asset_volumes = np.array([(l, up_volumes_hl[hour_index, l]) for l in available_asset_indeces if compatible_array[l, market_index] == 1])
    else:
        asset_volumes = np.array([(l, down_volumes_hl[hour_index, l]) for l in available_asset_indeces if compatible_array[l, market_index] == 1])
    
        
    if market_directions_m[market_index] == 0: # both
        total_up_volume = np.sum([asset[1] for asset in asset_volumes])
        total_down_volume = np.sum([asset[2] for asset in asset_volumes])
        if total_down_volume >= 1 and total_up_volume >= 1:
            #print("total volume is less than minimum volume")
            return asset_volumes[:][0], min(total_up_volume, total_down_volume)
        else:
            return [], 0
    else:
        total_volume = np.sum([asset[1] for asset in asset_volumes])
        if total_volume >= 1:
            return asset_volumes[:][0], total_volume
        else:
            #print("total volume is less than minimum volume")
            return []
        
    
def normalize_weather_data(weather_data, scaler = sklearn.preprocessing.MinMaxScaler()):
    data = weather_data.copy()
    data["precipitation"] = scaler.fit_transform(data[["precipitation"]])
    data["air_temp"] = scaler.fit_transform(data[["air_temp"]])
    return data



@numba.jit(nopython=True)
def get_features(hour_index : int, available_assets : np.array(int), market_index : int, expected_prices : np.array(int), expected_volumes : np.array(int), air_temp_array : np.array(int), precipitation_array : np.array(int), wind_speed_array : np.array(int), cloud_cover_array : np.array(int), da_array  : np.array(int), market_directions_m : np.array(int)):
    """ Function to get the features for the given hour and market. It is important to use features that will help the model learn which actions to take 
        and update the weights correctly for the given state.
        The features that possibly can be used here are the following:
        - day of week
        - hour of day
        - number of possible assets
        - Day Ahead (DA) price
        - Weather forecast
        - Market historical prices
        - Market historical volumes
        - frequency data (historical)
        
    Args:
        available_assets (new_meters.PowerMeter]): _description_
        hour (pd.Timestamp): _description_
        market (final_markets.ReserveMarket): _description_
        norm_exp_price_dict (dict) : dictionary of the normalized expected prices for each direction and area in the bid_timeframe 
        norm_exp_vol_dict (dict) : of the normalized expected volumes for each direction and area in the bid_timeframe 
        norm_w_df (pd.DataFrame) : normalized weather data 
        norm_da_df (pd.DataFrame) : normalized day ahead prices 
        L ([new_meters.PowerMeter]) : list of all possible assets

    Returns:
        _type_: _description_
    """

    market_direction = market_directions_m[market_index]
    day_of_week = hour_index // 24 #elif 2 if hour_index < 48 elif 3 if hour_index < 72 elif 4 if hour_index < 96 elif 5 if hour_index < 120 elif 6 if hour_index < 144 else 0
    hour_of_day = hour_index if hour_index < 24 else hour_index - 24
    expected_price = expected_prices[market_direction, hour_index] # må sjekke
    expected_volume = expected_volumes[market_direction, hour_index] # må teste
    precipitation = precipitation_array[day_of_week]
    temperature = air_temp_array[hour_index]
    wind_speed = wind_speed_array[hour_index]
    cloud_cover = cloud_cover_array[day_of_week]
    da_price = da_array[hour_index]
    
    
    return np.array(day_of_week/7, hour_of_day/24, expected_price, expected_volume, len(available_assets)/len(L), precipitation, temperature, wind_speed, cloud_cover, da_price, market_index/6)

@numba.jit(nopython=True)
def get_income_for_portfolio(volume : float, market_index : int,  hour_index : int, market_prices_hm : np.array(int)):
    """function to calculate the income for a given portfolio

    Args:
        portfolio ([new_meters.PowerMeter]): the portfolio for which the income is to be calculated
        market (final_markets.ReserveMarket): the market for which the income is to be calculated
        hour (pd.Timestamp): the hour for which the income is to be calculated

    Returns:
        float: the income for the given portfolio
    """
    
    return volume * market_prices_hm[hour_index, market_index]


@numba.jit(nopython=True)
def make_bid(market_index : int, hour_index : int, action : int, possible_assets : np.array(int), market_prices_hm, compatible_array : np.array(int), up_volumes_hl : np.array(int), down_volumes_hl : np.array(int), market_directions_m : np.array(int), top_n : int = 100, min_n : int = 20, iterations : int = 100):
    """ Function to actually place a bid

    Args:
        market_index (int): [description]
        hour_index (int): [description]
        action (int): [description]
        possible_assets (np.array): [description]
        market_prices_hm ([type]): [description]
        compatible_array (np.array): [description]
        up_volumes_hl (np.array): [description]
        down_volumes_hl (np.array): [description]
        market_directions_m (np.array): [description]
        top_n (int, optional): [description]. Defaults to 100.
        min_n (int, optional): [description]. Defaults to 20.
        iterations (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """
    if action == 0:
        return [], 0, 0
    elif action == 4:
        #portfolio, volume = get_feasible_portfolio_for_market(possible_assets = possible_assets, market = market, hour = hour)
        portfolio, volume = get_max_portfolio_for_market(available_asset_indeces= possible_assets, market_index = market_index, hour_index = hour_index, compatible_array = compatible_array, up_volumes_hl = up_volumes_hl, down_volumes_hl = down_volumes_hl, market_directions_m = market_directions_m)
        return portfolio, get_income_for_portfolio(volume, market_index, hour_index, market_prices_hm), volume
    else:
        sorting = ["min", "max", "mean"][action-1]
        portfolio, volume = get_n_portfolios_for_market(available_asset_indeces= possible_assets, market_index = market_index, hour_index = hour_index, compatible_array = compatible_array, up_volumes_hl = up_volumes_hl, down_volumes_hl = down_volumes_hl, market_directions_m = market_directions_m, sorting = sorting, top_n = top_n, min_n = min_n, iterations=iterations)
        return portfolio, get_income_for_portfolio(volume, market_index, hour_index, market_prices_hm), volume
    

def get_market_count_df(x : dict, y: dict, w : dict,  H : [pd.Timestamp], L : [new_meters.PowerMeter], M : [final_markets.ReserveMarket], Ir_hlm, Ia_hlm, Fu_h_l, Fd_h_l):
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
    
    df = pd.DataFrame(columns=["Market", "Hour", "Asset Count","Total Flex Volume [MWh]", "Total capacity revenue [EUR]", "Total activation revenue [EUR]"])
    for h, hour in enumerate(H):
        for m, market in enumerate(M):
            if y[h, m].X > 0:
                amount_of_assets = sum(x[h, l, m].X for l in range(len(L)))
                capacity_income = sum(x[h, l, m].X * Ir_hlm[h, l, m] for l in range(len(L)))
                activation_income = sum(x[h, l, m].X * Ia_hlm[h, l, m] * w[h,m].X for l in range(len(L)))
                if market.direction == "up":
                    total_flex_volume = sum(x[h, l, m].X * Fu_h_l[h,l] for l in range(len(L)))
                elif market.direction == "down":
                    total_flex_volume = sum(x[h, l, m].X * Fd_h_l[h,l] for l in range(len(L)))
                else:
                    total_flex_volume = 0
                    for l, load in enumerate(L):
                        if load.direction == "up":
                            total_flex_volume += x[h, l, m].X * Fu_h_l[h,l]
                        elif load.direction == "down":
                            total_flex_volume += x[h, l, m].X * Fd_h_l[h,l]
                        else:
                            total_flex_volume += min(x[h, l, m].X * Fu_h_l[h,l], x[h, l, m].X * Fd_h_l[h,l])
                           
                df.loc[len(df)] = [market.name, hour, amount_of_assets, total_flex_volume, capacity_income, activation_income]
    return df

@numba.jit(nopython=True)
def initialize_weights(n_features :int , n_actions : int, zeros : bool = False):
    """ Function to initialize the weights to use in the RL model

    Args:
        n_features (int): number of features
        n_actions (int): number of actions
        zeros (bool, optional): If True, the weights will be initialized to zeros. Defaults to False which means that the weights will be initialized to random values between 0 and 0.1.

    Returns:
        np.array([np.array()]): the weights in a nested array of shape (n_actions, n_features)
    """
    if zeros:
        return np.array([np.zeros((n_features)) for _ in range(n_actions)])
    else:
        return np.array([np.random.normal(0, 0.1, n_features) for _ in range(n_actions)])


def initialize_weights(n_features :int , n_actions : int):
    return [np.zeros((n_features)) for _ in range(n_actions)]

