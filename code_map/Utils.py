from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
from functools import reduce
import pytz
from code_map import final_markets, new_meters, timeframes
import gurobipy as gp
import pickle
import math



# will have to add a constraint for timeframe
def get_frequency_data(tf : timeframes.TimeFrame, freq_directory : str):
    """ Function to get frequency data for a given time frame. The data is for each 0.1 second.

    Args:
        tf (TimeFrame): the wanted timeframe where the data is wanted
        freq_directory (str): relative path to the directory where the frequency data is stored

    Returns:
        df: dataframe of the frequency data for the given time frame 
    """
    
    freq_files_list = [file for file in os.listdir(freq_directory) if file.endswith('.csv')]
    freq_dfs = []

    for file in freq_files_list:
        file_path = os.path.join(freq_directory, file)
        data = pd.read_csv(file_path)
        freq_dfs.append(data)
        
    freq_df = pd.concat(freq_dfs, ignore_index= True)
    format = "%Y-%m-%d %H:%M:%S.%f"
    freq_df["Time"] = pd.to_datetime(freq_df["Time"], format = format) 
    freq_df["Time"] = freq_df["Time"].dt.tz_localize("Europe/Oslo")
    start_datetime = pd.Timestamp(year = tf.year, month= tf.start_month, day=tf.start_day, hour= tf.start_hour, tz = "Europe/Oslo") #Europe/Oslo    
    end_datetime = pd.Timestamp(year = tf.year, month= tf.end_month, day=tf.end_day, hour= tf.end_hour, tz = "Europe/Oslo") + pd.Timedelta(hours=1)
     # Create a copy of the slice to avoid the warning when using in-place operations
    filtered_df = freq_df[(freq_df["Time"] >= start_datetime) & (freq_df["Time"] <= end_datetime)].copy()

    # Can use in-place operations on filtered_df
    filtered_df.sort_values(by="Time", inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)
    
    return filtered_df

#freq_data = get_frequency_data(timeframes.one_day, '../master-data/frequency_data/2023-06')

#freq_data.head()

def find_frequency_quarters(freq_df : pd.DataFrame, hours : [pd.Timestamp], index = True):
    average_freq_dict = {}
    for h, hour in enumerate(hours):
        start_datetime = hour 
        quarter_1 = hour + pd.Timedelta(minutes=15) # 0-15 minutes
        quarter_2 = quarter_1 + pd.Timedelta(minutes=15) # 15-30 minutes
        quarter_3 = quarter_2 + pd.Timedelta(minutes=15) # 30-45 minutes
        quarter_4 = quarter_3 + pd.Timedelta(minutes=15) # 45-60 minutes
        quarter_tfs = [start_datetime, quarter_1, quarter_2, quarter_3, quarter_4]
        # make 4 quarters for each hour within a loop
        average_freqs = []  
        for i in range(len(quarter_tfs)-1):
            filtered_df = freq_df[(freq_df["Time"] >= quarter_tfs[i]) & (freq_df["Time"] <= quarter_tfs[i+1])]
            mean_val = filtered_df["Value"].mean()
            if mean_val < 49.9:
                mean_val = 49.9
            elif mean_val > 50.1:
                mean_val = 50.1
            average_freqs.append(mean_val)
        if index:
            average_freq_dict[h] = average_freqs        
        else:
            average_freq_dict[hour] = average_freqs
    return average_freq_dict
    
"""tf = timeframes.one_day
hours = pd.date_range(start = pd.Timestamp(year = tf.year, month= tf.start_month, day=tf.start_day, hour= tf.start_hour, tz = "Europe/Oslo"), end = pd.Timestamp(year = tf.year, month= tf.end_month, day=tf.end_day, hour= tf.end_hour, tz = "Europe/Oslo"), freq = 'H', tz = "Europe/Oslo")

average_freq_dict = find_frequency_quarters(freq_df = freq_data, hours = hours)

average_freq_dict[0]
sum(average_freq_dict[0])

10 * 2 * (200 - sum(average_freq_dict[0]))

2000 * 2 -10* 2* sum(average_freq_dict[0])"""


def get_FCR_N_percentages(freq_df : pd.DataFrame, hours : timeframes.TimeFrame, markets : [final_markets.ReserveMarket]):

    """ Get a dictionary of the activation percentages for each market and hour. 
    The activation percentages are based on the frequency data and says how much of the time the frequency was above 50.0 Hz and below 50.0 Hz for each hour

    Args:
        freq_df (pd.DataFrame): dataframe of the frequency data for the given time frame fetched from get_frequency_data
        timeframe (list): list of the wanted hours
        markets (list): list of the markets

    Returns:
        dict: a dictionary of the activation percentages for each market and hour
    """
    
    freq_dict = {}
    for h, hour in enumerate(hours):
        start_datetime = hour 
        end_datetime = hour + pd.Timedelta(hours=1)
        for m, market in enumerate(markets):
            filtered_df = freq_df[(freq_df["Time"] >= start_datetime) & (freq_df["Time"] <= end_datetime)]
            if "FCR_N" in market.name:
                FCR_N_up_activation = filtered_df.loc[(filtered_df["Value"] > 49.9) & (filtered_df["Value"] < 50.0)]
                FCR_N_down_activation = filtered_df.loc[(filtered_df["Value"] < 50.1) & (filtered_df["Value"] > 50.0)]
                freq_dict[h,m] = (len(FCR_N_up_activation)/len(filtered_df), len(FCR_N_down_activation)/len(filtered_df))
            else:
                freq_dict[h,m] = (0,0)
    return freq_dict



def get_afrr_activation_data(tf : timeframes.TimeFrame, afrr_directory : str, direction : str):
    """
    Get a dataframe of the activation volumes for afrr up or down for each hour in the timeframe

    Args:
        tf (TimeFrame): the wanted timeframe where the data is wanted
        afrr_directory (str): relative path to the directory where the afrr data is stored
        direction (str): either "Up" or "Down" depending on which direction of afrr is wanted

    Returns:
        pd.DataFrame: a dataframe of the activation volumes for afrr up or down for each hour in the timeframe
    """
    afrr_files_list = [file for file in os.listdir(afrr_directory) if file.endswith('.csv')]
    afrr_dfs = []
    start_string = 'Regulation ' + direction + ' Activated'
    #print(start_string)

    for file in afrr_files_list:
        file_path = os.path.join(afrr_directory, file)
        data = pd.read_csv(file_path)
        data.rename(columns = {"Balancing Time Unit (Automatic Frequency Restoration Reserve (aFRR))" : "Time"}, inplace = True)
        data = data.loc[data["Source"] == "Not specified"]
        #print(data.columns)
        # Create a list of columns to be dropped
        columns_to_drop = [col for col in data.columns if not col.startswith(start_string) and not col.startswith("Time")]
        # Drop the columns from the DataFrame
        data = data.drop(columns=columns_to_drop)
        #print(data.columns)
        #data.drop(columns = ["Source"], inplace = True)
        afrr_dfs.append(data)
        
    afrr_dfs = [df.set_index('Time') for df in afrr_dfs]
    afrr_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), afrr_dfs)
    afrr_df = afrr_df.reset_index()

    afrr_df["Time"] = afrr_df["Time"].str.slice(0, 16)
    
    DST_TRANSITION = '2023-03-26 02:00:00'
    dst_transition_datetime = pd.to_datetime(DST_TRANSITION)
    
    
    afrr_df["Time"] = pd.to_datetime(afrr_df["Time"], format = "%d.%m.%Y %H:%M")
    
    # Adjust the non-existent times
    afrr_df.loc[(afrr_df["Time"] >= dst_transition_datetime) & 
                (afrr_df["Time"] < dst_transition_datetime + pd.Timedelta(hours=1)), 
                "Time"] += pd.Timedelta(hours=1)
    timezone = pytz.timezone("Europe/Oslo")
    afrr_df["Time"] = afrr_df["Time"].apply(lambda x: timezone.localize(x, is_dst=None))
    
    #afrr_df["Time"] = afrr_df["Time"].dt.tz_localize("Europe/Oslo", ambiguous='infer')
    start_datetime = pd.Timestamp(year = tf.year, month= tf.start_month, day=tf.start_day, hour= tf.start_hour, tz = "Europe/Oslo") #Europe/Oslo    
    end_datetime = pd.Timestamp(year = tf.year, month= tf.end_month, day=tf.end_day, hour= tf.end_hour, tz = "Europe/Oslo")
    filtered_df = afrr_df[(afrr_df["Time"] >= start_datetime) & (afrr_df["Time"] <= end_datetime)]
    
    filtered_df.sort_values(by = "Time", inplace = True)
    filtered_df.reset_index(inplace = True, drop = True)
    filtered_df.iloc[:, 1:6] = filtered_df.iloc[:, 1:6].astype(float)
    
    return filtered_df

"""afrr_activation = get_afrr_activation_data(tf = one_day, afrr_directory = '../master-data/aFRR_activation/', direction = "Up")
afrr_activation.columns
afrr_activation[afrr_activation.columns[1:6]]
#afrr_activation.to_csv("only_up_activations.csv")

afrr_activation.iloc[:, 1:6] = afrr_activation.iloc[:, 1:6].astype(float)

afrr_activation.columns"""



def get_timestamps(tf : timeframes.TimeFrame):
    """ Function to get timestamps for a given time frame

    Args:
        tf (TimeFrame): the wanted timeframe where the data is wanted

    Returns:
        list: list of timestamps for the given time frame
    """
    start_datetime = pd.Timestamp(year = tf.year, month= tf.start_month, day=tf.start_day, hour= tf.start_hour, tz = "Europe/Oslo") #Europe/Oslo    
    end_datetime = pd.Timestamp(year = tf.year, month= tf.end_month, day=tf.end_day, hour= tf.end_hour, tz = "Europe/Oslo")
    timestamps = pd.date_range(start = start_datetime, end = end_datetime, freq = 'H', tz = "Europe/Oslo")
    return timestamps


def get_all_sets(timeframe : timeframes.TimeFrame, areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]):
    """ Function to get all the sets needed for the optimization problem

    Args:
        timeframe (TimeFrame): the wanted timeframe where the data is wanted

    Returns:
        L (list) : list of powermeter objects with the data for each meter within the timeframe
        M (list) : list of reservemarket objects with the data for each market within the timeframe
        F (dict) : dictionary of the activation percentages for each market and hour due to the frequency data
        H (list) : list of timestamps for the given timeframe
        freq_data (pd.DataFrame) : dataframe of the frequency data for the given timeframe
        power_meter_dict (dict) : dictionary of the powermeter's consumption data for each hour within the timeframe
        consumption_data (pd.DataFrame) : dataframe of all the meters' consumption data
    """
    #FCR DIRECTORIES
    fcr_d_1_directory = "../master-data/markets-data/FCR_D-1-2023.xlsx"
    fcr_d_2_directory = "../master-data/markets-data/FCR_D-2-2023.xlsx"

    # aFRR DIRECTORIES
    afrr_up_directory = '../master-data/markets-data/aFFR/up_2023'
    afrr_down_directory = '../master-data/markets-data/aFFR/down_2023'

    # RK DIRECTORIES
    rk_price_down_path = "../master-data/markets-data/RK/new_rk_price_down.csv"
    rk_price_up_path = "../master-data/markets-data/RK/new_rk_price_up.csv"
    rk_volume_up_path = "../master-data/markets-data/RK/new_rk_vol_up.csv"
    rk_volume_down_path = "../master-data/markets-data/RK/new_rk_vol_down.csv"

    # RKOM DIRECTORIES
    rkom_2022_path = "../master-data/markets-data/RKOM.xlsx"
    rkom_2023_path = "../master-data/markets-data/Rkom-2023.xlsx"

    #SPOT PRICE DIRECTORY
    spot_path = "../master-data/spot_data/spot_june_23.csv"

    # CATEGORY DIRECTORIES
    cat_path_list = ["../master-data/categorization_data/harktech_meters.csv",  "../master-data/categorization_data/ev_meters.csv"]
    
    consumption_data =pd.read_csv('../master-data/customers-data/added_type_and_comp.csv')
    all_market_list = final_markets.get_market_list(tf = timeframe, spot_path=spot_path, fcr_d_1_path= fcr_d_1_directory, fcr_d_2_path=fcr_d_2_directory, afrr_up_directory=afrr_up_directory, afrr_down_directory=afrr_down_directory, rk_price_down_path=rk_price_down_path,rk_price_up_path= rk_price_up_path, rk_volume_up_path=rk_volume_up_path, rk_volume_down_path=rk_volume_down_path, rkom_22_path=rkom_2022_path, rkom_23_path= rkom_2023_path, areas=areas)
    power_meter_dict = new_meters.create_meter_objects(consumption_data = consumption_data, tf= timeframe, reference_tf= timeframes.one_month, category_path_list=cat_path_list, areas = areas) 
    freq_data = get_frequency_data(tf = timeframe, freq_directory= '../master-data/frequency_data/2023-06')
    
    H = get_timestamps(tf = timeframe)

    # Define the sets
    L = list(power_meter_dict.values())  # List of PowerMeter objects
    M = all_market_list  # List of ReserveMarket objects

    F = get_FCR_N_percentages(freq_df = freq_data, hours = H , markets = M)
    
    
    return L, M, F, H, freq_data, power_meter_dict, consumption_data

def get_parameters(L : [new_meters.PowerMeter], M : [final_markets.ReserveMarket], H : [pd.Timestamp]):
    """ function to return the sets needed for the optimization problem

    Args:
        L (list(PowerMeter)): list of powermeter objects with the data for each meter within the timeframe
        M (list(ReserveMarket)): list of reservemarket objects with the data for each market within the timeframe
        H (list(pd.TimeStamp)): list of hourly timestamps within the timeframe

    Returns:
        L_u (list(PowerMeter)): list of powermeter objects with the data for each meter within the timeframe that have direction up or both
        L_d (list(PowerMeter)): list of powermeter objects with the data for each meter within the timeframe that have direction down or both
        Fu_h_l (matrix(float)): The flex volume up for each hour and load
        Fd_h_l (matrix(float)): The flex volume down for each hour and load
        R_h_l (matrix(float)): The response time for each hour and load
        P_h_m (matrix(float)): The price for each hour and market
        Vp_h_m (matrix(float)): The volume for each hour and market
        Vm_m (list(float)): The min volume for each market
        R_m (list(float)): The response time for each market
    """
    # make a list of only the meters that have direction up or both
    L_u = [meter for meter in L if meter.direction != 'down']
    L_d = [meter for meter in L if meter.direction != 'up']

    Fu_h_l = np.array([[load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] if load.direction != "down" else 0 for load in L] for hour in H]) # set of flex volumes for meters, if load.direction != "down"
    Fd_h_l = np.array([[load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] if load.direction != "up" else 0 for load in L] for hour in H]) # set of flex volumes for meters, if load.direction != "up"

    R_h_l = np.array([[load.response_time for load in L]] * len(H)) # set of response times for meters

    P_h_m = np.array([[market.price_data.loc[market.price_data["Time(Local)"] == hour].values[0][1] for market in M] for hour in H]) # set of prices for markets
    Vp_h_m = np.array([[market.volume_data.loc[market.volume_data["Time(Local)"] == hour].values[0][1] for market in M] for hour in H]) # set of volumes for markets

    Vm_m = [market.min_volume for market in M] # set of min values for markets
    R_m = [market.response_time for market in M] # set of response times for markets
    return L_u, L_d, Fu_h_l, Fd_h_l, R_h_l, P_h_m, Vp_h_m, Vm_m, R_m

def get_dominant_direction(freq_df : pd.DataFrame, hour : pd.Timestamp):
    """will find out which direction is dominant within an hour

    Args:
        freq_data (pd.DataFrame): dataframe of the frequency data
        hour (pd.Timestamp): the wanted hour
    """
    start_datetime = hour 
    end_datetime = hour + pd.Timedelta(hours=1)
        
    filtered_df = freq_df[(freq_df["Time"] >= start_datetime) & (freq_df["Time"] <= end_datetime)]
    #print(filtered_df)
    avg_freq = filtered_df["Value"].mean()
    #print(avg_freq)
    if avg_freq > 50.0:
        return "up"
    else:
        return "down"
    
def get_income_dictionaries(H : [pd.Timestamp], M : [final_markets.ReserveMarket], L : [new_meters.PowerMeter], freq_data : pd.DataFrame, Fu_h_l : np.array, Fd_h_l : np.array, P_h_m : np.array, Vp_h_m : np.array, F : dict, markets_dict : dict, timeframe : timeframes.TimeFrame, areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]):
    """ Function to get the income dictionaries for the optimization problem

    Args:
        H (list(pd.TimeStamp)): list of hourly timestamps within the timeframe
        M (list(ReserveMarket)): list of reservemarket objects with the data for each market within the timeframe
        L (list(PowerMeter)): list of powermeter objects with the data for each meter within the timeframe
        freq_data (pd.DataFrame): dataframe of the frequency data
        Fu_h_l (matrix(float)): The flex volume up for each hour and load
        Fd_h_l (matrix(float)): The flex volume down for each hour and load
        P_h_m (matrix(float)): The price for each hour and market
        Vp_h_m (matrix(float)): The volume for each hour and market
        F (dict): dictionary for frequency data
        markets_dict (dict): dictionary of the markets
        timeframe (TimeFrame): TimeFrame object which tells the timeframe
        areas (list(str)): list of the areas. Defaults to ["NO1", "NO2", "NO3", "NO4", "NO5"]. Only include one of the areas in the list if one is wanted

    Returns:
        Ir_hlm (dict): The income from the reserve markets for each hour, load, and market
        Ia_hlm (dict): The income from the activation markets for each hour, load, and market
        Va_hm (dict): The volume from the activation markets for each hour and market
    """
    afrr_activation_up = get_afrr_activation_data(tf = timeframe, afrr_directory = '../master-data/aFRR_activation/', direction = "Up") #  a dataframe of the activation volumes for afrr up for each hour in the timeframe
    afrr_activation_down = get_afrr_activation_data(tf = timeframe, afrr_directory = '../master-data/aFRR_activation/', direction = "Down") #  a dataframe of the activation volumes for afrr down for each hour in the timeframe
    frequency_quarter_dict = find_frequency_quarters(freq_df = freq_data, hours = H, index = True) # a dictionary of the frequency quarters for each hour
    
    
    Ir_hlm = {} # reservation income
    Ia_hlm = {} # activation income
    Va_hm = {} # activation volume

    # Precompute values that can be determined outside the inner loop
    RK_up_prices = {}
    RK_down_prices = {}
    aFRR_activation_up_volume = {}
    aFRR_activation_down_volume = {}
    for area in areas:
        for hour in H:
            RK_up_prices[(area, hour)] = markets_dict["RK_up_" + area].price_data.loc[markets_dict["RK_up_" + area].price_data["Time(Local)"] == hour].values[0][1]
            RK_down_prices[(area, hour)] =  markets_dict["RK_down_" + area].price_data.loc[markets_dict["RK_down_" + area].price_data["Time(Local)"] == hour].values[0][1]
            col_name_up = [col for col in afrr_activation_up.columns if area in col][0]
            aFRR_activation_up_volume[(area, hour)] = afrr_activation_up[col_name_up].loc[afrr_activation_up["Time"] == hour].values[0] 
            col_name_down = [col for col in afrr_activation_down.columns if area in col][0]
            aFRR_activation_down_volume[(area, hour)] = afrr_activation_down[col_name_down].loc[afrr_activation_down["Time"] == hour].values[0]
    
    for h, hour in enumerate(H):
        for m, market in enumerate(M):
            for l, load in enumerate(L):
                if market.direction == "both": #only accounts for FCR-N
                    if load.direction == "up":
                        Ir_hlm[h,l,m] = Fu_h_l[h,l] * P_h_m[h,m]
                    elif load.direction == "down":
                        Ir_hlm[h,l ,m] = Fd_h_l[h,l] * P_h_m[h,m]
                    else:
                        up_val, down_val = F[h,m]
                        ## the reservation price should only depend on which direction has the lowst aggregated volume. 
                        # FCR-N should have bids with equal volume in both directions and therefore it depends on the lowest volume of the portfolio
                        Ir_hlm[h,l,m] = (Fu_h_l[h,l] * up_val + Fd_h_l[h,l] * down_val) * P_h_m[h,m] 
                elif market.direction == "up":
                    Ir_hlm[h,l,m] = Fu_h_l[h,l] * P_h_m[h,m] if load.direction != "down" else 0
                else: # market.direction == "down"
                    Ir_hlm[h,l,m] = Fd_h_l[h,l] * P_h_m[h,m] if load.direction != "up" else 0
                if market.capacity_market: 
                    if "FCR_N" in market.name:
                        #up_val, down_val = F[h,m]
                        Va_hm[h,m] = Vp_h_m[h,m] #* (up_val + down_val) if (up_val + down_val) > 0 else 0
                        frequency_quarters = frequency_quarter_dict[h]
                        if load.direction == "both":
                            activation_vol = (Fu_h_l[h,l] * up_val + Fd_h_l[h,l] * down_val)
                            # Add to the objective expression
                        elif load.direction == "up":
                            activation_vol = Fu_h_l[h,l]
                        else: # load.direction == "down"
                            activation_vol = Fd_h_l[h,l]
                        activated_FCR = 2000 * activation_vol - 10 * activation_vol * sum(frequency_quarters) # can this give nan value?
                        if activated_FCR < 0:
                            activated_FCR = activated_FCR * -1
                            Ia_hlm[h,l,m] = activated_FCR * RK_down_prices[(market.area, hour)] # nan values here?
                        else:
                            Ia_hlm[h,l,m] = activated_FCR * RK_up_prices[(market.area, hour)] # nan values here?
                        
                    elif "aFRR" in market.name: # will have to add the other markets later - especially aFRR and RKOM
                        if market.direction == "up":
                            activated_volume = aFRR_activation_up_volume[(market.area, hour)]
                            Va_hm[h,m] = activated_volume
                            if  load.direction != "down" and activated_volume > 0:
                                Ia_hlm[h,l,m] = Fu_h_l[h,l] * RK_up_prices[(market.area, hour)]
                            else:
                                Ia_hlm[h,l,m] = 0
                        elif market.direction == "down": 
                            activated_volume = aFRR_activation_down_volume[(market.area, hour)]
                            Va_hm[h,m] = activated_volume
                            if load.direction != "up" and activated_volume > 0:
                                Ia_hlm[h,l,m] = Fd_h_l[h,l] * RK_down_prices[(market.area, hour)]
                            else:
                                Ia_hlm[h,l,m] = 0
                        else:
                            Ia_hlm[h,l,m] = 0
                        
                    else: # No activation income in other markets than afrr and fcr-d, just regular income
                        Ia_hlm[h,l,m] = 0
                        Va_hm[h,m] = 0
                else:
                    # No capacity market, just regular income
                    Ia_hlm[h,l,m] = 0
                    Va_hm[h,m] = 0
    return Ir_hlm, Ia_hlm, Va_hm

def get_compatibility_dict(L : [new_meters.PowerMeter], M : [final_markets.ReserveMarket], index = True):
    """ function to get a dict of compatible markets for each asset. The dict is either a dict of lists of indexes or a dict of lists of objects.
        The function checks the response time, the area and the direction of the asset and the market to see if they are compatible

    Args:
        H (list(pd.TimeStamp)): list of hourly timestamps within the timeframe
        L (list(PowerMeter)): list of powermeter objects with the data for each meter within the timeframe
        M (list(ReserveMarket)): list of reservemarket objects with the data for each market within the timeframe

    Returns:
        dict: list of compatible loads for each market
    """
    compatible_dict = {}


    for m, market in enumerate(M):
        asset_list = []
        for l, asset in enumerate(L):
            if asset.response_time <= market.response_time and market.area == asset.area:
                if market.direction == "up":
                    if asset.direction != "down":
                        asset_list.append(l) if index else asset_list.append(asset)
                elif market.direction == "down":
                    if asset.direction != "up":
                        asset_list.append(l) if index else asset_list.append(asset)
                else:
                    asset_list.append(l) if index else asset_list.append(asset)
        if index:
            compatible_dict[m] = asset_list
        else:
            compatible_dict[market] = asset_list
    return compatible_dict

def run_optimization_model(L : [new_meters.PowerMeter], M : [final_markets.ReserveMarket], H : [pd.Timestamp], F : dict, Ir_hlm : dict, Ia_hlm : dict, Va_hm : dict, Vp_h_m : np.array, Vm_m : list, R_m : list, R_h_l : np.array, Fu_h_l : np.array, Fd_h_l : np.array, compatible_list : dict, log_filename : str, model_name : str):
    """ Function to create and run an optimization model for bidding in the reserve markets for a given set of meters and markets. The bidding is for historical data

    Args:
        L (list(new_meters.PowerMeter]): set of all meters
        M (list(final_markets.ReserveMarket]): set of all markets
        H (list(pd.Timestamp]): set of all hours
        F (dict): Dictionary to find the activation percentages for each market and hour
        Ir_hlm (dict): Dictionary to find the reservation income from the reserve markets for each hour, load, and market
        Ia_hlm (dict): Dictionary to find the activation income from the markets for each hour, load, and market
        Va_hm (dict): Dictionary to find the activation volume from the markets for each hour and market
        Vp_h_m (np.array): The volume for each hour and market
        Vm_m (list): Minimum volume for each market
        R_m (list): Response time for each market
        R_h_l (np.array): Response time for each load each hour
        Fu_h_l (np.array): Up flex volume for each load that are compatible with up markets for each hour 
        Fd_h_l (np.array): Down flex volume for each load that are compatible with down markets for each hour
        dominant_directions (list): list of dominant directions for each hour
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
    # Create a new model
    model = gp.Model(model_name)
    model.setParam('OutputFlag', 1)
    model.setParam('LogFile', log_filename)

    # Create decision variables
    x = {}
    d = {}
    y = {}
    w = {}
    for h in range(len(H)):
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
    model.setObjective(sum(x[h,l,m] * (Ir_hlm[h,l,m] + Ia_hlm[h,l,m] * w[h,m]) for h in range(len(H)) for l in range(len(L)) for m in range(len(M))), gp.GRB.MAXIMIZE) # can possibly remove the x on the activation income

    # Add constraints
    for h in range(len(H)):
        for l in range(len(L)):
            # Each asset can only be connected to one market at a time
            model.addConstr(sum(x[h, l, m] for m in range(len(M))) <= 1, f"single_market_for_asset_at_hour_{h}_nr.{l}")
        
        for m, market in enumerate(M):
            up_val, down_val = F[h,m]
            if up_val + down_val > 0:
                model.addConstr(w[h,m] <= y[h,m], f"market_{m}_can_not_be_activated_at_hour_{h}_if_it_is_not_active")
            else:
                model.addConstr(w[h,m] == 0, f"market_{m}_can_not_be_activated_at_hour_{h}_if_it_is_not_active")
            
            # Connect the binary variables by using big M
            model.addConstr(sum(x[h, l, m] for l in range(len(L))) <= len(L) * y[h, m], f"asset_connection_for_hour_{h}_market_{m}")
        
            # Max volume constraint
            
            if market.direction == "up":
                # capacity volume constraint
                model.addConstr(sum(x[h, l, m] * Fu_h_l[h,l] for l in range(len(L))) <= Vp_h_m[h,m]  * y[h,m], f"max_volume_for_hour_{h}_market_{m}")
                # activation volume constraint
                model.addConstr(sum(x[h, l, m] * Fu_h_l[h,l] for l in range(len(L))) * w[h,m] <= Va_hm[h,m], f"max_volume_for_activation_in-_market_{m}_at_hour_{h}")
                # min volume capacity constraint
                model.addConstr(sum(x[h, l, m] * Fu_h_l[h,l] for l in range(len(L))) >= Vm_m[m] * y[h, m], f"min_volume_for_hour_{h}_market_{m}") 

            elif market.direction == "down":
                # max capacity volume constraint
                model.addConstr(sum(x[h, l, m] * Fd_h_l[h,l] for l in range(len(L))) <= Vp_h_m[h,m]  * y[h,m], f"max_volume_for_hour_{h}_market_{m}")
                # max activation volume constraint
                model.addConstr(sum(x[h, l, m] * Fd_h_l[h,l] for l in range(len(L))) * w[h,m] <= Va_hm[h,m], f"max_volume_for_activation_in_market_{m}_at_hour_{h}")
                # min volume capacity constraint
                model.addConstr(sum(x[h, l, m] * Fd_h_l[h,l] for l in range(len(L))) >= Vm_m[m] * y[h, m], f"min_volume_for_hour_{h}_market_{m}") 

            else: # market.direction == "both" => In FCR-N you must be able to activate in both directions
                # max capacity volume constraint
                model.addConstr(sum(x[h, l, m] * Fu_h_l[h,l] for l in range(len(L))) <= Vp_h_m[h,m]  * y[h,m], f"max_volume_for_hour_{h}_market_{m}")
                # max activation volume constraint
                model.addConstr(sum(x[h, l, m] * Fu_h_l[h,l] for l in range(len(L))) * w[h,m] <= Va_hm[h,m] , f"max_volume_for_activation_in_market_{m}_at_hour_{h}")
                # min capacity volume constraint
                model.addConstr(sum(x[h, l, m] * Fu_h_l[h,l] for l in range(len(L))) >= Vm_m[m] * y[h, m], f"min_volume_for_hour_{h}_market_{m}") 
                
                # max capacity volume constraint
                model.addConstr(sum(x[h, l, m] * Fd_h_l[h,l] for l in range(len(L))) <= Vp_h_m[h,m] * y[h,m], f"max_volume_for_hour_{h}_market_{m}")
                # max activation volume constraint
                model.addConstr(sum(x[h, l, m] * Fd_h_l[h,l] for l in range(len(L))) * w[h,m] <= Va_hm[h,m], f"max_volume_for_activation_in_market_{m}_at_hour_{h}")
                # min capacity volume constraint
                model.addConstr(sum(x[h, l, m] * Fd_h_l[h,l] for l in range(len(L))) >= Vm_m[m] * y[h, m], f"min_volume_for_hour_{h}_market_{m}") 
            
                # add a constraint where the reserved volume in the FCR_N markets has to be the same for each direction
                # I cant find a way to do this in a mathematical model, i think it should be held out of the equation for this part and rather use it in the dynamic model
                #model.addConstr(sum(x[h, l, m] * Fu_h_l[h,l] for l in range(len(L))) == sum(x[h, l, m] * Fd_h_l[h,l] for l in range(len(L)))) # this can be hard as they have to be exactly equal
               

            # The response times for loads l connected to market m cannot exceed the max response time for m
            for l in range(len(L)):
                model.addConstr(x[h,l,m] * R_h_l[h,l] <= R_m[m] * y[h,m], f"response_time_for_hour_{h}_market_{m}")
                
    model.optimize(callback_factory(log_filename))

    if model.status == gp.GRB.Status.INFEASIBLE:
        model.computeIIS()
        
        
    return model, x, y, w, d   

def callback_factory(log_filename):
    def optimized_callback(model, where):
        with open(log_filename, 'a') as log_file:
            if where == gp.GRB.Callback.MIP:
                objbst = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
                objbnd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
                if objbst < gp.GRB.INFINITY:
                    log_entry = f"Current best objective: {objbst}, Best bound: {objbnd}\n"
                    print(log_entry, end='')
                    log_file.write(log_entry)
                    log_file.flush()

            elif where == gp.GRB.Callback.MESSAGE:
                msg = model.cbGet(gp.GRB.Callback.MSG_STRING)
                print(msg.strip(), end='')
                log_file.write(msg)
                log_file.flush()
    return optimized_callback


def run_optimization_model_no_numpy(L : [new_meters.PowerMeter], M : [final_markets.ReserveMarket], H : [pd.Timestamp], F : dict, Ir_hlm : dict, Ia_hlm : dict, Va_hm : dict, Vm_m : list, compatible_list : dict, log_filename : str, model_name : str):
    """ Function to create and run an optimization model for bidding in the reserve markets for a given set of meters and markets. The bidding is for historical data

    Args:
        L (list(new_meters.PowerMeter]): set of all meters
        M (list(final_markets.ReserveMarket]): set of all markets
        H (list(pd.Timestamp]): set of all hours
        F (dict): Dictionary to find the activation percentages for each market and hour
        Ir_hlm (dict): Dictionary to find the reservation income from the reserve markets for each hour, load, and market
        Ia_hlm (dict): Dictionary to find the activation income from the markets for each hour, load, and market
        Va_hm (dict): Dictionary to find the activation volume from the markets for each hour and market
        Vm_m (list): Minimum volume for each market
        R_m (list): Response time for each market
        dominant_directions (list): list of dominant directions for each hour
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
    # Create a new model
    model = gp.Model(model_name)
    # Create decision variables
    x = {}
    d = {}
    y = {}
    w = {}
    for h in range(len(H)):
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
    model.setObjective(sum(x[h,l,m] * (Ir_hlm[h,l,m] + Ia_hlm[h,l,m] * w[h,m]) for h in range(len(H)) for l in range(len(L)) for m in range(len(M))), gp.GRB.MAXIMIZE) # can possibly remove the x on the activation income

    # Add constraints
    for h, hour in enumerate(H):
        for l in range(len(L)):
            # Each asset can only be connected to one market at a time
            model.addConstr(sum(x[h, l, m] for m in range(len(M))) <= 1, f"single_market_for_asset_at_hour_{h}_nr.{l}")
        
        for m, market in enumerate(M):
            up_val, down_val = F[h,m]
            if up_val + down_val > 0:
                model.addConstr(w[h,m] <= y[h,m], f"market_{m}_can_not_be_activated_at_hour_{h}_if_it_is_not_active")
            else:
                model.addConstr(w[h,m] == 0, f"market_{m}_can_not_be_activated_at_hour_{h}_if_it_is_not_active")
            
            # Connect the binary variables by using big M
            model.addConstr(sum(x[h, l, m] for l in range(len(L))) <= len(L) * y[h, m], f"asset_connection_for_hour_{h}_market_{m}")
        
            # Max volume constraint
            
            if market.direction == "up":
                # capacity volume constraint
                model.addConstr(sum(x[h, l, m] * load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "down") <= market.volume_data.loc[market.volume_data["Time(Local)"] == hour].values[0][1] * y[h,m], f"max_volume_for_hour_{h}_market_{m}")
                # activation volume constraint
                model.addConstr(sum(x[h, l, m] * load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "down")* w[h,m] <= Va_hm[h,m], f"max_volume_for_activation_in-_market_{m}_at_hour_{h}")
                # min volume capacity constraint
                model.addConstr(sum(x[h, l, m] * load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "down") >= Vm_m[m] * y[h, m], f"min_volume_for_hour_{h}_market_{m}") 

            elif market.direction == "down":
                # max capacity volume constraint
                model.addConstr(sum(x[h, l, m] * load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "up") <= market.volume_data.loc[market.volume_data["Time(Local)"] == hour].values[0][1] * y[h,m], f"max_volume_for_hour_{h}_market_{m}")
                # max activation volume constraint
                model.addConstr(sum(x[h, l, m] * load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "up") * w[h,m] <= Va_hm[h,m], f"max_volume_for_activation_in_market_{m}_at_hour_{h}")
                # min volume capacity constraint
                model.addConstr(sum(x[h, l, m] * load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "up") >= Vm_m[m] * y[h, m], f"min_volume_for_hour_{h}_market_{m}") 

            else: # market.direction == "both" => In FCR-N you must be able to activate in both directions
                # capacity volume constraint
                model.addConstr(sum(x[h, l, m] * load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "down") <= market.volume_data.loc[market.volume_data["Time(Local)"] == hour].values[0][1] * y[h,m], f"max_volume_for_hour_{h}_market_{m}")
                # activation volume constraint
                model.addConstr(sum(x[h, l, m] * load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "down")* w[h,m] <= Va_hm[h,m], f"max_volume_for_activation_in-_market_{m}_at_hour_{h}")
                # min volume capacity constraint
                model.addConstr(sum(x[h, l, m] * load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "down") >= Vm_m[m] * y[h, m], f"min_volume_for_hour_{h}_market_{m}") 

                # max capacity volume constraint
                model.addConstr(sum(x[h, l, m] * load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "up") <= market.volume_data.loc[market.volume_data["Time(Local)"] == hour].values[0][1] * y[h,m], f"max_volume_for_hour_{h}_market_{m}")
                # max activation volume constraint
                model.addConstr(sum(x[h, l, m] * load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "up") * w[h,m] <= Va_hm[h,m], f"max_volume_for_activation_in_market_{m}_at_hour_{h}")
                # min volume capacity constraint
                model.addConstr(sum(x[h, l, m] * load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "up") >= Vm_m[m] * y[h, m], f"min_volume_for_hour_{h}_market_{m}") 
            
                # add a constraint where the reserved volume in the FCR_N markets has to be the same for each direction
                # I cant find a way to do this in a mathematical model, i think it should be held out of the equation for this part and rather use it in the dynamic model
                #model.addConstr(sum(x[h, l, m] * Fu_h_l[h,l] for l in range(len(L))) == sum(x[h, l, m] * Fd_h_l[h,l] for l in range(len(L)))) # this can be hard as they have to be exactly equal
               

            # The response times for loads l connected to market m cannot exceed the max response time for m
            for l, load in enumerate(L):
                model.addConstr(x[h,l,m] * load.response_time <= market.response_time * y[h,m], f"response_time_for_hour_{h}_market_{m}")
                
    # Enable logging
    model.setParam('LogFile', log_filename)

    # Solve the model
    model.optimize()
        
    if model.status == gp.GRB.Status.INFEASIBLE:
        model.computeIIS()
        
        
    return model, x, y, w, d  


def run_batched_optimization_model(L : [new_meters.PowerMeter], M : [final_markets.ReserveMarket], H : [pd.Timestamp], F : dict, freq_data :pd.DataFrame, P_h_m : np.array, Vp_h_m : np.array, Vm_m : list, R_m : list, R_h_l : np.array, Fu_h_l : np.array, Fd_h_l : np.array, compatible_list : dict, log_filename : str, model_name : str):
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
        tf = timeframes.TimeFrame(year = 2023, start_month = 6, end_month = 6, start_day = batch_H[0].day, end_day = batch_H[0].day, start_hour = 0, end_hour = 23)

        # the income
        batch_Ir_hlm, batch_Ia_hlm, batch_Va_hm = get_income_dictionaries(H = batch_H, M = M, L =L, freq_data = freq_data, Fu_h_l = batch_Fu_h_l, Fd_h_l = batch_Fd_h_l, P_h_m = batch_P_h_m, Vp_h_m = batch_Vp_h_m, F =F, markets_dict = market_name_dict, timeframe = tf)
       

        # Run the optimization model for this batch
        _, x, y, w, d = run_optimization_model(L= L, M= M, H = batch_H,F= F, Ir_hlm= batch_Ir_hlm, Ia_hlm= batch_Ia_hlm, Va_hm= batch_Va_hm, Vp_h_m= batch_Vp_h_m, Vm_m=Vm_m, R_m=R_m, R_h_l=batch_R_h_l, Fu_h_l=batch_Fu_h_l, Fd_h_l=batch_Fd_h_l, compatible_list=compatible_list, log_filename=log_filename, model_name=f"{model_name}_batch_{b}")
        # Store results
        #aggregated_results['models'].append(model)
        aggregated_results['x_values'].append(x)
        aggregated_results['y_values'].append(y)
        aggregated_results['w_values'].append(w)
        aggregated_results['d_values'].append(d)
        test_solution_validity(x, y, w, batch_Va_hm, L, M, batch_H, F)


    # Process aggregated_results as needed
    return aggregated_results

def test_solution_validity(x : dict, y : dict, w : dict, Va_hm : dict, L : [new_meters.PowerMeter], M : [final_markets.ReserveMarket], H : [pd.Timestamp], F : dict):
    """ function to test the validity of the solution provided by a solver

    Args:
        x (dict): dictionary of the binary variable which tells if an asset is connected to a market
        y (dict): dictionary of the binary variable which tells if a market has any bids
        w (dict): dictionary of the binary variable which tells if a market is activated
        Va_hm (dict): dictionary of the volume from the activation markets for each hour and market
        L (list(PowerMeter)): list of powermeter objects with the data for each meter within the timeframe
        M (list(ReserveMarket)): list of reservemarket objects with the data for each market within the timeframe
        H (list(pd.TimeStamp)): list of hourly timestamps within the timeframe
        dominant_directions (list(str)): list of the dominant direction for each hour
        F (dict): dictionary for frequency data
    Returns:
        str : a string that tells if the solution is valid. If not valid, the function will raise an error
    """
    for h, hour in enumerate(H):
        for l, load in enumerate(L):
            # Each asset can only be connected to one market at a time
            assert sum(x[h, l, m].X for m in range(len(M))) <= 1, f"Asset {l} connected to multiple markets at hour {h}"
            for m, market in enumerate(M):
                # Directionality constraints
                if load.direction == "up" and market.direction == "down":
                    assert x[h, l, m].X == 0, f"Up-direction asset {l} connected to down-direction market {m} at hour {h}"
                elif load.direction == "down" and market.direction == "up":
                    assert x[h, l, m].X == 0, f"Down-direction asset {l} connected to up-direction market {m} at hour {h}"
                #elif market.direction == "both" and load.direction != "both":
                    #assert x[h, l, m].X == 0, f"Asset {l} with specific direction connected to both-direction market {m} at hour {h}"
                elif market.area != load.area:
                    assert x[h, l, m].X == 0, f"Asset {l} in area {load.area} connected to market {m} in area {market.area} at hour {h}"
                
                # Response time constraints
                assert x[h, l, m].X * load.response_time <= market.response_time * y[h, m].X, f"Asset {l} connected to market {m} at hour {h} violates response time constraint"
                
        for m, market in enumerate(M):
            # Connect the binary variables by using big M
            assert sum(x[h, l, m].X for l in range(len(L))) <= len(L) * y[h, m].X, f"More than allowed assets connected at hour {h} to market {m}"

            #total_flex_volume = sum(x[h, l, m].X * load.flex_volume["value"].loc[load.flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L))

            # Min volume constraint
            if market.direction == "up":
                total_flex_volume = sum(x[h, l, m].X * load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "down")
            elif market.direction == "down":
                total_flex_volume = sum(x[h, l, m].X * load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "up")
            else: # direction = "both"
                total_flex_volume =min(sum(x[h, l, m].X * load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "down"), sum(x[h, l, m].X * load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "up"))
            
            assert round(total_flex_volume, 5) >= market.min_volume * y[h, m].X, f"Minimum volume constraint violated at hour {h} for market {m}"
            
            # Max volume constraint for both capacity and activation
            if market.direction == "up":
                total_max_volume = sum(x[h, l, m].X * load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "down")

            elif market.direction == "down":
                total_max_volume = sum(x[h, l, m].X * load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "up")

            else:
                """if dominant_directions[h] == "up":
                    total_max_volume = sum(x[h, l, m].X * load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "down")
                else:
                    total_max_volume = sum(x[h, l, m].X * load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "up")"""
                total_up_max_volume = sum(x[h, l, m].X * load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "down")
                total_down_max_volume = sum(x[h, l, m].X * load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "up")
                up_frac, down_frac = F[h,m]
                total_max_volume = (total_up_max_volume * up_frac + total_down_max_volume * down_frac)
            
             # Assert the constraints
            activation_constraint = round(total_max_volume, 5)  * w[h,m].X <= Va_hm[h,m]
            assert activation_constraint, f"Activation constraint violated for hour {h}, market {m}"
            market_max_volume = market.volume_data.loc[market.volume_data["Time(Local)"] == hour].values[0][1]
            assert total_max_volume <= market_max_volume * y[h,m].X, f"Maximum volume constraint violated at hour {h} for market {m}"
    return "Solution is valid"

def get_market_count_dict(x : dict, H : [pd.Timestamp], L : [new_meters.PowerMeter], M : [final_markets.ReserveMarket], dominant_directions):
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
                if x[h, l, m] > 0.5:
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
                total_flex_volume = sum(x[h, l, m] * load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "down")
            elif market.direction == "down":
                total_flex_volume = sum(x[h, l, m] * load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "up")
            else:
                if dominant_directions[h] == "up":
                    total_flex_volume = sum(x[h, l, m] * load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "down")
                else:
                    total_flex_volume = sum(x[h, l, m] * load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "up")
                
            flex_volumes.append(total_flex_volume)
        market_count["Total Flex Volume"] = flex_volumes
        market_count_dict[hour] = market_count
    return market_count_dict

def compare_current_and_new_solution(current_x_pkl_file : str, new_x_values : dict, H : [pd.Timestamp], L : [new_meters.PowerMeter], M : [final_markets.ReserveMarket], dominant_directions : [str]):
    """ Function to compare the results of the optimization problem for the new solution and an old one which is stored as a pkl file

    Args:
        current_x_pkl_file (str): Filename of the pkl file with the old solution
        new_x_values (dict): The new x values
        H (pd.Timestamp]): set of all hours
        L (new_meters.PowerMeter]): set of all meters
        M (final_markets.ReserveMarket]): set of all markets
        dominant_directions (str]): list of the dominant direction for each hour

    Returns:
        dict: dict of the differences of the two solutions for each hour. Each hour holds two dataframes, one for the old solution and one for the new solution
    """
    # Load the saved values
    with open(current_x_pkl_file, 'rb') as f:
        original_x_values = pickle.load(f)

    old_dict = get_market_count_dict(x = original_x_values, H = H, L = L, M= M, dominant_directions= dominant_directions)
    new_dict = get_market_count_dict(x = new_x_values, H=H, L=L, M=M, dominant_directions= dominant_directions)

    differences = {}
    for key in old_dict:
        if not old_dict[key].equals(new_dict[key]):
            differences[key] = (new_dict[key], old_dict[key])
            

    for key, (mod_val, orig_val) in differences.items():
        print(f"Difference for hour {key}: \n Original={display(orig_val)}, \n  Modified={display(mod_val)}")
    return differences
