import gurobipy as gp
import pandas as pd
from code_map import final_markets, new_meters, utils, weather, timeframes
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import random
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def random_arg_max(possible_actions):
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

def greedy_action(possible_actions : list, epsilon : float ):
    """returns the index of the greedy action

    Args:
        possible_actions (lsit): list of the possible actions
        epsilon (float): float number between 0 and 1, often close to 0

    Returns:
        int: index of the greedy action
    """
    if random.random() <= (1- epsilon): # pick greedy
        return random_arg_max(possible_actions)
    else:
        return random.randint(0, len(possible_actions)-1) # random

def get_expected_prices_and_volumes_dict(bid_timeframe : [pd.Timestamp], markets : [final_markets.ReserveMarket]):
    """ function to calculate the expected prices for each direction and area in the bid_timeframe

    Args:
        bid_timeframe (_type_): _description_
        markets (_type_): _description_

    Returns:
        _type_: _description_
    """
    expected_prices = {}
    expected_volumes = {}
    for directions in ["up", "down", "both"]:
        #for area in ["NO1", "NO2", "NO3", "NO4", "NO5"]:
        area = "NO5"
        for hour in bid_timeframe:
            if np.isnan(np.mean([market.price_data.loc[market.price_data["Time(Local)"] == hour].values[0][1] for market in markets if market.area == area and market.direction == directions])):
                print("nan")
                print([market.price_data.loc[market.price_data["Time(Local)"] == hour].values[0][1] for market in markets if market.area == area and market.direction == directions])
                print(f"directions: {directions}, area: {area}, hour: {hour}")
            expected_prices[(directions, area, hour)] = np.mean([market.price_data.loc[market.price_data["Time(Local)"] == hour].values[0][1] for market in markets if market.area == area and market.direction == directions])
            if np.isnan(np.mean([market.volume_data.loc[market.volume_data["Time(Local)"] == hour].values[0][1] for market in markets if market.area == area and market.direction == directions])):
                print("nan")
                print([market.volume_data.loc[market.volume_data["Time(Local)"] == hour].values[0][1] for market in markets if market.area == area and market.direction == directions])
                print(f"directions: {directions}, area: {area}, hour: {hour}")
            expected_volumes[(directions, area, hour)] = np.mean([market.volume_data.loc[market.volume_data["Time(Local)"] == hour].values[0][1] for market in markets if market.area == area and market.direction == directions])
    return expected_prices, expected_volumes

def normalize_dict_vals(dict : dict, norm_method :str = "min_max" ) -> dict:
    """Function to normalize the values of a dictionary

    Args:
        dict (dict): [description]
        norm_method (str): Normalization method; can choose between min-max normalization or z-score normalization. Defaults to "min_max".

    Returns:
        dict: the same dictionary as input, but with normalized values
    """
    if norm_method == "min_max":
        values = list(dict.values())
        min_value = min(values)
        max_value = max(values)

        # Normalize and update the dictionary
        normalized_dict = {k: (v - min_value) / (max_value - min_value) for k, v in dict.items()}
    else:
        import statistics
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        # Normalize and update the dictionary
        normalized_dict = {k: (v - mean) / std for k, v in dict.items()}

    return normalized_dict


 
def get_possible_dates(date : pd.Timestamp):
    """ Function to get the possible dates for placing a bid given the current date

    Args:
        date (pd.Timestamp): the current date

    Returns:
        (pd.date_range, str): the possible dates for placing a bid and for which market
    """
    if date.hour == 17: # FCR D-2
        return (pd.date_range(date + timedelta(days=1) + timedelta(hours=7), date + timedelta(days = 2) + timedelta(hours = 6), freq='H', tz = "Europe/Oslo"), "D_2")
    elif date.hour == 7: # aFRR
        return (pd.date_range(date + timedelta(hours = 17), date + timedelta(days = 1) + timedelta(hours = 16), freq='H', tz = "Europe/Oslo"), "aFRR")
    elif date.hour == 18: # FCR D-1
        return (pd.date_range(date + timedelta(hours=6), date + timedelta(days = 1) + timedelta(hours = 5), freq='H', tz = "Europe/Oslo"), "D_1")
    else:
        return ([], "No bids")
    
def get_feasible_portfolio_for_market(possible_assets : [new_meters.PowerMeter], market : final_markets.ReserveMarket, hour : pd.Timestamp, compatible_dict : dict):
    """This function will return one feasible combination quickly.
        DISCLAIMER : This function doesnt take procured volume in the given market in to account.

    Args:
        possible_assets ([new_meters.PowerMeter]): list of possible assets for the given market and given hour
        market (final_markets.ReserveMarket): the market for which the portfolio is to be found
        hour (pd.Timestamp): the hour for which the portfolio is to be found
        compatible_dict (dict): dictionary of compatible assets for each market

    Returns:
        list(new_meters.PowerMeter): the feasible portfolio that will be used for the given market in the given hour
    """
   
    # Fetch volumes for each asset at the given hour
    feasible_assets = [asset for asset in possible_assets if asset in compatible_dict[market]] # exclude assets that are not compatible with the given market
    # Fetch volumes for each asset at the given hour
    asset_volumes = []
    for asset in feasible_assets:
        if market.direction == "both":
            if asset.direction == "both":
                vol = min(asset.up_flex_volume["value"].loc[asset.up_flex_volume["Time(Local)"] == hour].values[0], asset.down_flex_volume["value"].loc[asset.down_flex_volume["Time(Local)"] == hour].values[0])
                asset_volumes.append((asset, vol))
            else:
                vol = asset.up_flex_volume["value"].loc[asset.up_flex_volume["Time(Local)"] == hour].values[0] if asset.direction == "up" else asset.down_flex_volume["value"].loc[asset.down_flex_volume["Time(Local)"] == hour].values[0]
                asset_volumes.append((asset, vol))
        else:
            vol = asset.up_flex_volume["value"].loc[asset.up_flex_volume["Time(Local)"] == hour].values[0] if market.direction == "up" else asset.down_flex_volume["value"].loc[asset.down_flex_volume["Time(Local)"] == hour].values[0]
            asset_volumes.append((asset, vol))
            

    # Sort assets by volume in descending order
    asset_volumes.sort(key=lambda x: x[1], reverse=True)

    # Find a feasible combination
    feasible_combination = []
    total_volume = 0
    for asset, volume in asset_volumes:
        feasible_combination.append(asset)
        total_volume += volume
        if total_volume >= market.min_volume:
            break
    
    if total_volume < market.min_volume:
        return [], 0

    return feasible_combination, total_volume

def get_n_portfolios_for_market(possible_assets : [new_meters.PowerMeter], market : final_markets.ReserveMarket, hour : pd.Timestamp, compatible_dict : dict, top_n=100, iterations=100):
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
    feasible_assets = [asset for asset in possible_assets if asset in compatible_dict[market]] # exclude assets that are not compatible with the given market
    # Fetch volumes for each asset at the given hour
    
    asset_volumes = []
    for asset in feasible_assets:
        if market.direction == "both":
            if asset.direction == "both":
                vol = min(asset.up_flex_volume["value"].loc[asset.up_flex_volume["Time(Local)"] == hour].values[0], asset.down_flex_volume["value"].loc[asset.down_flex_volume["Time(Local)"] == hour].values[0])
                asset_volumes.append((asset, vol))
            else:
                vol = asset.up_flex_volume["value"].loc[asset.up_flex_volume["Time(Local)"] == hour].values[0] if asset.direction == "up" else asset.down_flex_volume["value"].loc[asset.down_flex_volume["Time(Local)"] == hour].values[0]
                asset_volumes.append((asset, vol))
        else:
            vol = asset.up_flex_volume["value"].loc[asset.up_flex_volume["Time(Local)"] == hour].values[0] if market.direction == "up" else asset.down_flex_volume["value"].loc[asset.down_flex_volume["Time(Local)"] == hour].values[0]
            asset_volumes.append((asset, vol))
            

    # Sort assets by volume in descending order
    asset_volumes.sort(key=lambda x: x[1], reverse=True)

    # Select top N assets
    top_assets = asset_volumes[:top_n]
    
    #feasible_combinations = set()
    feasible_combinations = {}
    if len(top_assets) > 0:
        for _ in range(iterations):
            # Randomly sample a smaller subset from top assets
            num_assets_to_sample = min(len(top_assets), random.randint(1, top_n))

            sampled_assets = random.sample(top_assets, k=num_assets_to_sample)

            # Greedy addition to meet minimum volume
            combination, total_volume = [], 0
            for asset, volume in sampled_assets:
                combination.append(asset)
                total_volume += volume
                if total_volume >= market.min_volume:
                    #feasible_combinations.add((tuple(combination), total_volume))
                    feasible_combinations[tuple(combination)] = total_volume
                    break
    
    return feasible_combinations


def get_max_portfolio_for_market(possible_assets : [new_meters.PowerMeter], market : final_markets.ReserveMarket, hour : pd.Timestamp, compatible_dict : dict):
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
    feasible_assets = [asset for asset in possible_assets if asset in compatible_dict[market]] # exclude assets that are not compatible with the given market
    # Fetch volumes for each asset at the given hour
    
    asset_volumes = []
    for asset in feasible_assets:
        if market.direction == "both":
            if asset.direction == "both":
                vol = min(asset.up_flex_volume["value"].loc[asset.up_flex_volume["Time(Local)"] == hour].values[0], asset.down_flex_volume["value"].loc[asset.down_flex_volume["Time(Local)"] == hour].values[0])
                asset_volumes.append(vol)
            else:
                vol = asset.up_flex_volume["value"].loc[asset.up_flex_volume["Time(Local)"] == hour].values[0] if asset.direction == "up" else asset.down_flex_volume["value"].loc[asset.down_flex_volume["Time(Local)"] == hour].values[0]
                asset_volumes.append(vol)
        else:
            vol = asset.up_flex_volume["value"].loc[asset.up_flex_volume["Time(Local)"] == hour].values[0] if market.direction == "up" else asset.down_flex_volume["value"].loc[asset.down_flex_volume["Time(Local)"] == hour].values[0]
            asset_volumes.append(vol)
        
    if sum(asset_volumes) >= market.min_volume:
        return feasible_assets, sum(asset_volumes)
    else:
        return [], 0
    
def normalize_weather_data(weather_data, scaler = sklearn.preprocessing.MinMaxScaler(), ):
    data = weather_data.copy()
    data["precipitation"] = scaler.fit_transform(data[["precipitation"]])
    data["air_temp"] = scaler.fit_transform(data[["air_temp"]])
    return data

def get_features(bid_hour : pd.Timestamp, available_assets : [new_meters.PowerMeter], market : final_markets.ReserveMarket, norm_exp_price_dict : dict, norm_exp_vol_dict : dict, norm_w_df : pd.DataFrame, norm_da_df : pd.DataFrame, L : [new_meters.PowerMeter]):
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
    day_of_week = bid_hour.weekday()
    hour_of_day = bid_hour.hour
    expected_price = norm_exp_price_dict[(market.direction, market.area, bid_hour)]
    expected_volume = norm_exp_vol_dict[(market.direction, market.area, bid_hour)]
    precipitation = norm_w_df["precipitation"].loc[norm_w_df["Time (Local)"] == bid_hour]
    temperature = norm_w_df["air_temp"].loc[norm_w_df["Time (Local)"] == bid_hour]
    da_price = norm_da_df["settlement"].loc[norm_da_df["Time(Local)"] == bid_hour]
    
    
    return np.array(day_of_week/7, hour_of_day/24, expected_price, expected_volume, len(available_assets)/len(L), precipitation.values[0], temperature.values[0], da_price.values[0])


def get_income_for_portfolio(volume : float, market : final_markets.ReserveMarket, hour : pd.Timestamp):
    """function to calculate the income for a given portfolio

    Args:
        portfolio ([new_meters.PowerMeter]): the portfolio for which the income is to be calculated
        market (final_markets.ReserveMarket): the market for which the income is to be calculated
        hour (pd.Timestamp): the hour for which the income is to be calculated

    Returns:
        float: the income for the given portfolio
    """
    
    return volume * market.price_data.loc[market.price_data["Time(Local)"] == hour].values[0][1]

def make_bid(market : final_markets.ReserveMarket, hour : pd.Timestamp, action : int, possible_assets : [new_meters.PowerMeter], compatible_dict : dict):
    """Function to make a bid to a given market in a given hour

    Args:
        market (final_markets.ReserveMarket): the market to bid to
        hour (pd.Timestamp): the hour to bid to
        action (int): the index of the portfolio to bid
        possible_assets ([new_meters.PowerMeter]): list of possible assets for the given market and given hour
                compatible_dict (dict): dictionary of compatible assets for each market


    Returns:
        portfolio (list(new_meters.PowerMeter)): the portfolio that was bid
        income (float): the income for the portfolio in the given market at the given hour
    """
    if action == 0:
        return [], 0, 0
    elif action == 1:
        portfolio_dict = get_n_portfolios_for_market(possible_assets = possible_assets, market = market, hour = hour, compatible_dict= compatible_dict, top_n=100, iterations=100)
        portfolio = list(portfolio_dict.keys())[0] if len(portfolio_dict.keys()) > 0 else []
        volume = portfolio_dict[portfolio] if len(portfolio) > 0 else 0
        return portfolio, get_income_for_portfolio(volume, market, hour), volume
    else:
        #portfolio, volume = get_feasible_portfolio_for_market(possible_assets = possible_assets, market = market, hour = hour)
        portfolio, volume = get_max_portfolio_for_market(possible_assets = possible_assets, market = market, hour = hour, compatible_dict= compatible_dict)
        return portfolio, get_income_for_portfolio(volume, market, hour), volume
    

<<<<<<< HEAD
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
=======
def initialize_weights(n_features :int , n_actions : int):
    return [np.zeros((n_features)) for _ in range(n_actions)]
>>>>>>> 87f793b74b2c8f4bca7853f5a8c4e88d08c0a54b

