from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
from zoneinfo import ZoneInfo
from functools import reduce
import pytz
from code_map import final_markets, new_meters

@dataclass
class GlobalVariables:
    year : int
    start_month : int 
    end_month : int 
    start_day : int 
    end_day : int 
    start_hour : int 
    end_hour : int
    


one_hour = GlobalVariables(year = 2023, start_month = 6, end_month = 6, start_day = 26, end_day = 26, start_hour = 15, end_hour = 16) # it may be possible to start from hour 14

one_day = GlobalVariables(year = 2023, start_month = 6, end_month = 6, start_day = 26, end_day = 26, start_hour = 0, end_hour = 23)

one_week = GlobalVariables(year = 2023, start_month = 6, end_month = 6, start_day = 19, end_day = 25, start_hour = 0, end_hour = 23)

half_month =  GlobalVariables(year = 2023, start_month = 6, end_month = 6, start_day = 14, end_day = 30, start_hour = 0, end_hour = 23)

one_month = GlobalVariables(year = 2023, start_month = 6, end_month = 6, start_day = 1, end_day = 30, start_hour = 0, end_hour = 23)


# will have to add a constraint for timeframe
def get_frequency_data(tf : GlobalVariables, freq_directory : str):
    """ Function to get frequency data for a given time frame

    Args:
        tf (GlobalVariables): the wanted timeframe where the data is wanted
        freq_directory (str): relative path to the directory where the frequency data is stored

    Returns:
        df: dataframe for the given time frame and frequency data
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
    end_datetime = pd.Timestamp(year = tf.year, month= tf.end_month, day=tf.end_day, hour= tf.end_hour, tz = "Europe/Oslo")
    filtered_df = freq_df[(freq_df["Time"] >= start_datetime) & (freq_df["Time"] <= end_datetime)]
    filtered_df.sort_values(by = "Time", inplace = True)
    filtered_df.reset_index(inplace = True, drop = True)
    
    return filtered_df

#freq_data = get_frequency_data(one_week, '../master-data/frequency_data/2023-06')

#freq_data.head()

def get_FCR_N_percentages(freq_df : pd.DataFrame, timeframe, markets):
    """ Get a dictionary of the activation percentages for each market and hour

    Args:
        freq_df (pd.DataFrame): dataframe of the frequency data
        timeframe (list): list of the wanted hours
        markets (list): list of the markets

    Returns:
        dict: a dictionary of the activation percentages for each market and hour
    """
    
    freq_dict = {}
    for h, hour in enumerate(timeframe):
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

#freq_df = get_frequency_data(tf = one_day, freq_directory = '../master-data/frequency_data/2023-06')
#freq_df.head()

def get_afrr_activation_data(tf : GlobalVariables, afrr_directory : str, direction : str):
    """
    Get a dataframe of the activation volumes for afrr up or down for each hour in the timeframe

    Args:
        tf (GlobalVariables): the wanted timeframe where the data is wanted
        afrr_directory (str): relative path to the directory where the afrr data is stored
        direction (str): either "Up" or "Down" depending on which direction of afrr is wanted

    Returns:
        pd.DataFrame: a dataframe of the activation volumes for afrr up or down for each hour in the timeframe
    """
    afrr_files_list = [file for file in os.listdir(afrr_directory) if file.endswith('.csv')]
    afrr_dfs = []
    start_string = 'Regulation ' + direction + ' Activated'
    print(start_string)

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



def get_timestamps(tf : GlobalVariables):
    """ Function to get timestamps for a given time frame

    Args:
        tf (GlobalVariables): the wanted timeframe where the data is wanted

    Returns:
        list: list of timestamps for the given time frame
    """
    start_datetime = pd.Timestamp(year = tf.year, month= tf.start_month, day=tf.start_day, hour= tf.start_hour, tz = "Europe/Oslo") #Europe/Oslo    
    end_datetime = pd.Timestamp(year = tf.year, month= tf.end_month, day=tf.end_day, hour= tf.end_hour, tz = "Europe/Oslo")
    timestamps = pd.date_range(start = start_datetime, end = end_datetime, freq = 'H', tz = "Europe/Oslo")
    return timestamps


def get_all_sets(timeframe : GlobalVariables):
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
    all_market_list = final_markets.get_market_list(tf = timeframe, spot_path=spot_path, fcr_d_1_path= fcr_d_1_directory, fcr_d_2_path=fcr_d_2_directory, afrr_up_directory=afrr_up_directory, afrr_down_directory=afrr_down_directory, rk_price_down_path=rk_price_down_path,rk_price_up_path= rk_price_up_path, rk_volume_up_path=rk_volume_up_path, rk_volume_down_path=rk_volume_down_path, rkom_22_path=rkom_2022_path, rkom_23_path= rkom_2023_path)
    power_meter_dict = new_meters.create_meter_objects(consumption_data = consumption_data, tf= timeframe, reference_tf= one_month, category_path_list=cat_path_list) 
    freq_data = get_frequency_data(timeframe, '../master-data/frequency_data/2023-06')
    
    H = get_timestamps(timeframe)

    # Define the sets
    L = list(power_meter_dict.values())  # List of PowerMeter objects
    M = all_market_list  # List of ReserveMarket objects

    F = get_FCR_N_percentages(freq_data, H, M)
    
    
    return L, M, F, H, freq_data, power_meter_dict, consumption_data

def get_parameters(L,M,H):
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
    
def get_income_dictionaries(H, M, L, dominant_directions, Fu_h_l, Fd_h_l, P_h_m, Vp_h_m, F, markets_dict, timeframe):
    afrr_activation_up = get_afrr_activation_data(tf = timeframe, afrr_directory = '../master-data/aFRR_activation/', direction = "Up")
    afrr_activation_down = get_afrr_activation_data(tf = timeframe, afrr_directory = '../master-data/aFRR_activation/', direction = "Down")
    
    Ir_hlm = {} # reservation income
    Ia_hlm = {} # activation income
    Va_hm = {} # activation volume

    # Precompute values that can be determined outside the inner loop
    RK_up_prices = {}
    RK_down_prices = {}
    aFRR_activation_up_volume = {}
    aFRR_activation_down_volume = {}
    for area in ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']:
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
                if market.direction == "both":
                    if load.direction == "both":
                        if dominant_directions[h] == "up":
                            Ir_hlm[h,l,m] = Fu_h_l[h,l] * P_h_m[h,m]
                        else:
                            Ir_hlm[h,l ,m] = Fd_h_l[h,l] * P_h_m[h,m]
                        #I[h,l,m] =(Fu_h_l[h,l]+ Fd_h_l[h,l])/2 * P_h_m[h,m]
                    else:
                        Ir_hlm[h,l,m] = 0
                elif market.direction == "up":
                    if load.direction != "down":
                        Ir_hlm[h,l,m] = Fu_h_l[h,l] * P_h_m[h,m]
                    else:
                        Ir_hlm[h,l,m] = 0
                else: # market.direction == "down"
                    if load.direction != "up":
                        Ir_hlm[h,l,m] = Fd_h_l[h,l] * P_h_m[h,m]
                    else:
                        Ir_hlm[h,l,m] = 0
                if market.capacity_market: 
                    if "FCR_N" in market.name:
                        up_val, down_val = F[h,m]
                        Va_hm[h,m] = Vp_h_m[h,m] * (up_val + down_val) if (up_val + down_val) > 0 else 0
                        if load.direction == "both":
                            activation_income = (Fu_h_l[h,l] * up_val * RK_up_prices[(market.area, hour)] + 
                                                Fd_h_l[h,l] * down_val * RK_down_prices[(market.area, hour)])
                            # Add to the objective expression
                            Ia_hlm[h,l,m] = activation_income
                        else:
                            Ia_hlm[h,l,m] = 0
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
                        
                    else: # No activation income, just regular income
                        Ia_hlm[h,l,m] = 0
                        Va_hm[h,m] = 0
                else:
                    # No capacity market, just regular income
                    Ir_hlm[h,l,m] = P_h_m[h,m] * Vp_h_m[h,m]
                    Ia_hlm[h,l,m] = 0
                    Va_hm[h,m] = 0
    return Ir_hlm, Ia_hlm, Va_hm

def get_compatibility_list(H,L,M):
    compatible_list = []
    for _ in H:
        hour_list = []
        for l, asset in enumerate(L):
            asset_list = []
            for m, market in enumerate(M):
                if asset.direction == "up":
                    if market.direction == "up":
                        if market.area == asset.area or market.area == "all":
                            asset_list.append(m)
                elif asset.direction == "down":
                    if market.area == asset.area or market.area == "all":
                        if market.direction == "down":
                            asset_list.append(m)
                    
                elif asset.direction == "both":
                    if market.area == asset.area  or market.area == "all":
                        asset_list.append(m)
            hour_list.append(asset_list)
        compatible_list.append(hour_list)
    return compatible_list


def test_solution_validity(x, y, w, Va_hm, L, M, H, dominant_directions, F,):
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
                elif market.direction == "both" and load.direction != "both":
                    assert x[h, l, m].X == 0, f"Asset {l} with specific direction connected to both-direction market {m} at hour {h}"
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
                if dominant_directions[h] == "up":
                    total_flex_volume = sum(x[h, l, m].X * load.up_flex_volume["value"].loc[load.up_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "down")
                else:
                    total_flex_volume = sum(x[h, l, m].X * load.down_flex_volume["value"].loc[load.down_flex_volume["Time(Local)"] == hour].values[0] for l, load in enumerate(L) if load.direction != "up")
            
            assert total_flex_volume >= market.min_volume * y[h, m].X, f"Minimum volume constraint violated at hour {h} for market {m}"
            
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
            activation_constraint = total_max_volume  * w[h,m].X <= Va_hm[h,m]
            assert activation_constraint, f"Activation constraint violated for hour {h}, market {m}"
            market_max_volume = market.volume_data.loc[market.volume_data["Time(Local)"] == hour].values[0][1]
            assert total_max_volume <= market_max_volume * y[h,m].X, f"Maximum volume constraint violated at hour {h} for market {m}"
    return "Solution is valid"

def get_market_count_dict(x, H, L, M, dominant_directions):
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
