import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar 
from datetime import datetime
import pytz
import openpyxl
from code_map import Inputs
from dataclasses import dataclass


"""
Tanker 07.11
Må ta hensyn til hvordan prising i RK fungerer. 
     hvis prisen er ulike spotpris for gitt time er utbetalting lik 0. 
     Det er kun differansen mellom spotpris og gitt pris som faktisk utbetales. 
     Dette er i utgangspunktet lagd for produksjonsanlegg (spesielt opp-markedene)
     Det vil altså si at jeg bør bruke RK-dataen for å finne activation price for de markedene hvor utbetalingen ved aktiveringen er RK
     Jeg kan ikke bruke RK pris datasettet utelukkende, men må lage en versjon hvor jeg finner differansen fra spotprisen i hver enkel time
     Må derfor laste ned spotpris data for å kunne finne denne differansen i hver time og gjøre dette datasettet lokalt.
    
Bør kanskje lage funksjoner slik at denne klassen bare fungerer som et interface og at jeg "lager" alt i opt_model filen
Bør kanskje også endre klassene til dataclasses
"""

@dataclass
class ReserveMarket:
    """
    Class to represent a reserve market.
    """
    name: str # name of the 
    response_time: int  # seconds
    duration: int  # minutes
    min_volume: float  # MW
    volume_data: pd.DataFrame # dataframe with columns "Time(Local)" and "Volume MW"
    price_data: pd.DataFrame # dataframe with columns "Time(Local)" and "Price EUR/MW"
    direction: str  # 'up', 'down', or 'both'
    area: str # NO1, NO2, NO3, NO4, NO5, or 'all'
    activation_price: pd.DataFrame = pd.DataFrame() # dataframe with columns "Time(Local)" and "Price EUR/MW"
    sleep_time: int = 60  # minutes, default is 60
    activation_threshold: float = 0  # frequency, default is 0
    capacity_market: bool = True  # indicates if the market is a capacity market, default is True


#________________________________Global variables____________________________________
"""version_variables = Inputs.one_day
year = version_variables.year
start_month = version_variables.start_month
end_month = version_variables.end_month
start_day = version_variables.start_day
end_day = version_variables.end_day
start_hour = version_variables.start_hour
end_hour = version_variables.end_hour"""
#area = version_variables.area
areas = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']
#___________________________________FFR_______________________________________
    
FFR_prof_hours = [23,24, 1, 2, 3, 4, 5, 6, 7]

def get_FFR_df(start_year, start_month, start_day, start_hour, end_year, end_month, end_day, end_hour):
    #timeframe = pd.date_range(start="2022-01-01 00:00:00", end="2023-10-01 00:00:00", freq="H", tz = "Europe/Oslo")
    timeframe = pd.date_range(start=pd.Timestamp(year= start_year, month= start_month, day = start_day, hour = start_hour), 
                              end= pd.Timestamp(year = end_year, month = end_month, day = end_day, hour = end_hour), freq="H", tz = "Europe/Oslo")
    ffr_df = pd.DataFrame(np.zeros((len(timeframe), 5)), columns= ["Time(Local)", "FFR-Flex Price [EUR/MW]", "FFR-Profil Price [EUR/MW]", "FFR-Flex Volume", "FFR-Profil Volume"])
    # The Time(Local) column should have datetime objects starting from 01.01.2022 until 01.01.2023 with hourly values
    ffr_df["Time(Local)"] = timeframe
    # The FFR-Flex Price [EUR/MW] column should be 0 until 29.04.2022 and then be equal to 450 until 30.10.2022 and then be 0 again
    ffr_df["FFR-Flex Price [EUR/MW]"] = 450 * 0.085
    ffr_df["FFR-Profil Price [EUR/MW]"] = 150 * 0.085
    for year in [start_year, end_year]:
        ffr_df["FFR-Flex Price [EUR/MW]"][(pd.Timestamp(year = year, month =10, day = 30, hour = 0, tz = "Europe/Oslo") < ffr_df["Time(Local)"]) & 
                                          ( ffr_df["Time(Local)"] < pd.Timestamp(year = year, month = 4, day = 29, hour = 0, tz = "Europe/Oslo"))]  = 0
        
        ffr_df["FFR-Profil Price [EUR/MW]"][(pd.Timestamp(year = year, month = 9, day = 3, hour = 0, tz = "Europe/Oslo") < ffr_df["Time(Local)"]) & 
                                            (ffr_df["Time(Local)"] < pd.Timestamp(year = year, month = 5, day = 27, hour = 0, tz = "Europe/Oslo"))]  = 0
        for date in ffr_df["Time(Local)"][(pd.Timestamp(year = year, month = 10, day = 30, hour = 0, tz = "Europe/Oslo") > ffr_df["Time(Local)"]) & 
                                          ( ffr_df["Time(Local)"] > pd.Timestamp(year = year, month = 4, day = 29, hour = 0, tz = "Europe/Oslo"))]:
            #print(date)
            if (date.hour > 6) & (date.hour < 22):
                ffr_df["FFR-Flex Price [EUR/MW]"].loc[(ffr_df["Time(Local)"] == date)] = 0
    
    return ffr_df


def create_FFR_markets(start_year, start_month, start_day, start_hour, end_year, end_month, end_day, end_hour):
    # opening_date= datetime.datetime(2022, 5, 27), end_date= datetime.datetime(2022,9,3)
    ffr_data = get_FFR_df(start_year = start_year, start_month = start_month, start_day = start_day, start_hour = start_hour, end_year = end_year, end_month = end_month, end_day = end_day, end_hour = end_hour)

    #print(ffr_data)


    FFR_profile = ReserveMarket(name = "FFR_profile", direction = "up", area = "all", response_time= 1.3, duration= 0.5, min_volume= 1, sleep_time=15, activation_threshold= 49.7, capacity_market= True, price_data= ffr_data.drop(columns= ["FFR-Flex Volume", "FFR-Flex Price [EUR/MW]", "FFR-Profil Volume"]), volume_data= ffr_data.drop(columns= ["FFR-Flex Price [EUR/MW]", "FFR-Flex Volume", "FFR-Profil Price [EUR/MW]"]))

    FFR_flex = ReserveMarket(name = "FFR_flex", direction = "up", area = "all", response_time= 1.3, duration= 0.5, min_volume= 5, sleep_time=15, activation_threshold= 49.7, price_data= ffr_data.drop(columns= ["FFR-Flex Volume", "FFR-Profil Price [EUR/MW]", "FFR-Profil Volume"]), volume_data= ffr_data.drop(columns= ["FFR-Flex Price [EUR/MW]", "FFR-Profil Volume", "FFR-Profil Price [EUR/MW]"]))

    return [FFR_flex, FFR_profile]


#___________________________________RK_________________________________________

"""rk_price_down_path = "../master-data/markets-data/RK/new_rk_price_down.csv"
rk_price_up_path = "../master-data/markets-data/RK/new_rk_price_up.csv"
rk_volume_up_path = "../master-data/markets-data/RK/new_rk_vol_up.csv"
rk_volume_down_path = "../master-data/markets-data/RK/new_rk_vol_down.csv"
spot_path = "../master-data/spot_data/spot_june_23.csv"""

def initialize_rk_data(price_down_path : str, price_up_path : str, volume_down_path: str, volume_up_path : str):
    rk_price_down = pd.read_csv(price_down_path)
    rk_price_up = pd.read_csv(price_up_path)
    rk_volume_up = pd.read_csv(volume_up_path)
    rk_volume_down = pd.read_csv(volume_down_path)

    rk_price_down.drop(columns = ["currency"], inplace=True)
    rk_price_up.drop(columns = ["currency"], inplace=True)

    return {"price_down" : rk_price_down,"price_up" : rk_price_up,"volume_up" : rk_volume_up,"volume_down" : rk_volume_down}

def preprocess_spot_data(df : pd.DataFrame, start_month : int, year : int, start_day : int, start_hour : int, end_hour : int, end_month : int, end_day : int, area : str):
    start_date = pd.Timestamp(year, start_month, start_day, start_hour).tz_localize('Europe/Oslo')
    end_date = pd.Timestamp(year, end_month, end_day, end_hour).tz_localize('Europe/Oslo')
   
    df["start_time"] = pd.to_datetime(df["start_time"])
 
    df["start_time"] = df["start_time"].dt.tz_convert('Europe/Oslo')
   
    df = df.loc[(df["start_time"] >= start_date) & (df["start_time"] <= end_date)]
    df.rename(columns={'start_time':'Time(Local)'}, inplace=True)
    df.sort_values(by=['Time(Local)', "delivery_area"], inplace=True)
    # remove duplicates
    df.drop_duplicates(subset=['Time(Local)', "delivery_area", "settlement"], inplace=True)
    return df.loc[df["delivery_area"] == area].reset_index(drop=True)

def preprocess_rk_dfs_dict(df_dict : dict, area : str, start_month : int, year : int, start_day : int, start_hour : int, end_hour : int, end_month : int, end_year : int, end_day : int, spot_path):
    start_datetime = pd.Timestamp(year = year, month= start_month, day=start_day, hour= start_hour, tz = "Europe/Oslo") #Europe/Oslo    
    end_datetime = pd.Timestamp(year = end_year, month= end_month, day=end_day, hour= end_hour, tz = "Europe/Oslo")
    updated_df_dict = {}
    spot_df = pd.read_csv(spot_path)
    updated_spot_df = preprocess_spot_data(spot_df, year = year, start_month = start_month, end_month = end_month, start_day = start_day, end_day = end_day, start_hour = start_hour, end_hour = end_hour, area = area)
    #print(updated_spot_df)
    for name in df_dict.keys():
        df = df_dict[name].copy()
        if name == "volume_down":
            # change from egative values to positive values
            df["value"].loc[df["value"] != 0] = df["value"].loc[df["value"] != 0] * -1
        df["start_time"] = pd.to_datetime(df["start_time"], format="%Y-%m-%d %H:%M:%S")
        df["start_time"] = df["start_time"].dt.tz_convert("Europe/Oslo")
        df.sort_values(by = ["start_time", "delivery_area"], inplace = True)
        df.rename(columns = {"start_time" : "Time(Local)"}, inplace = True)
        filtered_df = df[(df["Time(Local)"] >= start_datetime) & (df["Time(Local)"] <= end_datetime) & (df["delivery_area"] == area)]
        filtered_df.sort_values(by = ["Time(Local)"], inplace = True)
        if name == "price_up" :
            filtered_df["value"] = np.float64(filtered_df["value"]) - np.float64(updated_spot_df["settlement"]) 
        if name == "price_down":
            filtered_df["value"] = np.float64(updated_spot_df["settlement"]) - np.float64(filtered_df["value"])
        
        updated_df_dict[name] = filtered_df
        
    return updated_df_dict


"""rk_dfs_dict = preprocess_rk_dfs_dict(initialize_rk_data(rk_price_down_path, rk_price_up_path, rk_volume_down_path, rk_volume_up_path), area = "NO1", start_month = start_month, year = year, start_day = start_day, start_hour = start_hour, end_hour = end_hour, end_month = end_month, end_year = year, end_day = end_day, spot_path = spot_path)

rk_dfs_dict["price_down"]"""

def create_rk_markets(spot_path :str, price_down_path : str, price_up_path : str, volume_down_path: str, volume_up_path : str, start_month : int, year : int, start_day : int, start_hour : int, end_hour : int, end_month : int,  end_day : int):
    rk_dfs_dict = initialize_rk_data(price_down_path = price_down_path, price_up_path = price_up_path, volume_down_path = volume_down_path, volume_up_path = volume_up_path)
    rk_dicts = []
    areas = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']

    for area in areas:
        rk_dicts.append(preprocess_rk_dfs_dict(df_dict=rk_dfs_dict, area=area, year= year, end_year = year, start_month = start_month, end_month = end_month, start_day = start_day, end_day = end_day, start_hour = start_hour, end_hour = end_hour, spot_path= spot_path))

    RK_up_markets = []
    RK_down_markets = []
    

    for rk_dict, area in zip(rk_dicts, areas):
        
        RK_up_markets.append(ReserveMarket(name = "RK_up_" + area, direction = "up", area = area, capacity_market= False, response_time=60*15, duration = 60, min_volume = 5 if area == "NO1" or area == "NO3" else 10, sleep_time=0,activation_threshold=0, price_data= rk_dict["price_up"], volume_data= rk_dict["volume_up"]))
        RK_down_markets.append(ReserveMarket(name = "RK_down_" + area, direction = "down", area = area, response_time=60*15, capacity_market=False,  duration = 60, min_volume = 5 if area == "NO1" or area == "NO3" else 10, sleep_time=0,activation_threshold=0, price_data= rk_dict["price_down"], volume_data= rk_dict["volume_down"]))
    return RK_up_markets, RK_down_markets


#_____________________________________FCR_____________________________________




def preprocess_FCR(df: pd.DataFrame, start_month : int, year : int, start_day : int, end_month : int, end_day : int, start_hour : int, end_hour : int):
    """ The datasets downloaded from Statnett is quite messy and needs some preprocessing. This function removes all the columns that has price in NOK/MW as they are only 0/NaN. It also fills all the NaN values in the columns Price EUR/MW with 0.

    Args:
        df (pd.DataFrame): The dataframe to be preprocessed

    Returns:
        df: preprocessed version of the input dataframe
    """
    # drop the columns that only includes nan values
    for col in df.columns:
        if "NOK" in col:
            df = df.drop(columns=[col])
    df["FCR-D Price EUR/MW"] = df["FCR-D Price EUR/MW"].fillna(0)
    df.drop(columns= 'Hournumber', inplace=True)
    date_format = '%d.%m.%Y %H:%M:%S %z'
    
    df["Time(Local)"] = pd.to_datetime(df["Time(Local)"], format=date_format)
    
    start_datetime = pd.Timestamp(year = year, month= start_month, day = start_day, hour = start_hour, tz = "Europe/Oslo")    
    end_datetime = pd.Timestamp(year = year, month=end_month, day = end_day, hour = end_hour, tz = "Europe/Oslo")

    # Filter based on date range
    filtered_df = df[(df["Time(Local)"] >= start_datetime) & (df["Time(Local)"] <= end_datetime)]
        
    #  check for missing hours in the dataset and insert the missing hour with the value equal to 0. Each hour should have 5 values.
    
    # make a list of the timestamps in the timeframe
    timeframe = pd.date_range(start=start_datetime, end=end_datetime, freq="H", tz = "Europe/Oslo")
    
    area_set = set(['NO1', 'NO2', 'NO3', 'NO4', 'NO5'])
    #print(len(filtered_df))
    #print(len(timeframe) * 5)
    if len(filtered_df) < len(timeframe) * 5:
        for hour in timeframe:
            if filtered_df.loc[filtered_df["Time(Local)"] == hour].shape[0] < 5:
                found_areas = filtered_df["Area"].loc[filtered_df["Time(Local)"] == hour].values
                #print(f"found_areas : {found_areas}")
                found_areas_set = set(found_areas)
                #print(f"found_areas_set : {found_areas_set} ")
                missing_area = area_set - found_areas_set
                #print(f"missing_area : {missing_area}")
                missing_row = pd.DataFrame([[hour, missing_area.pop() , 0, 0, 0, 0]], columns=filtered_df.columns)
                filtered_df = filtered_df.append(missing_row, ignore_index=True)
    
    return filtered_df

def get_FCR_N_activation_income(freq_df : pd.DataFrame, tf : Inputs.GlobalVariables, rk_price_data : pd.DataFrame, area : str ):
    
    H = Inputs.get_timestamps(tf)
    
    activation_df = pd.DataFrame(np.zeros((len(tf.timeframe), 3)), columns= ["Time(Local)", "FCR-N up activation income", "FCR_N down activation income"])

    activation_df["Time(Local)"] = H
    for hour in H:
        start_datetime = hour 
        end_datetime = hour + pd.Timedelta(hours=1)
        
        filtered_df = freq_df[(freq_df["Time"] >= start_datetime) & (freq_df["Time"] <= end_datetime)]
        FCR_N_up_activation = filtered_df.loc[(filtered_df["Value"] > 49.9) & (filtered_df["Value"] < 50.0)]
        FCR_N_down_activation = filtered_df.loc[(filtered_df["Value"] < 50.1) & (filtered_df["Value"] > 50.0)]
        up_frac = len(FCR_N_up_activation)/len(filtered_df)
        down_frac = len(FCR_N_down_activation)/len(filtered_df)
        
        up_activation_income = up_frac * rk_price_data["value"].loc[rk_price_data["Time(Local)"] == hour].values[0][1] 
        down_activation_income = down_frac * rk_price_data.loc[rk_price_data["RK_down_" + area].price_data["Time(Local)"] == hour].values[0][1] 
        activation_df["FCR-N up activation income"].loc[activation_df["Time(Local)"] == hour] = up_activation_income
        activation_df["FCR_N down activation income"].loc[activation_df["Time(Local)"] == hour] = down_activation_income
    
    return activation_df

tf = Inputs.one_day

freq_data = Inputs.get_frequency_data(tf, '../master-data/frequency_data/2023-06')
rk_price_down_path = "../master-data/markets-data/RK/new_rk_price_down.csv"
rk_price_up_path = "../master-data/markets-data/RK/new_rk_price_up.csv"
rk_volume_up_path = "../master-data/markets-data/RK/new_rk_vol_up.csv"
rk_volume_down_path = "../master-data/markets-data/RK/new_rk_vol_down.csv"
spot_path = "../master-data/spot_data/spot_june_23.csv"

RK_down_markets , RK_up_markets = create_rk_markets(spot_path = spot_path, price_down_path = rk_price_down_path, price_up_path = rk_price_up_path, volume_down_path = rk_volume_down_path, volume_up_path = rk_volume_up_path, start_month = tf.start_month, year = tf.year, start_day = tf.start_day, start_hour = tf.start_hour, end_hour = tf.end_hour, end_month = tf.end_month, end_day = tf.end_day)
    
RK_down_markets[0].price_data

def get_area_FCR_df(filtered_df : pd.DataFrame, area : str):
    # Filter by area
    monthly_area_df = filtered_df[filtered_df["Area"] == area]
        
    # Sort by "Time(Local)" column
    monthly_area_df = monthly_area_df.sort_values(by="Time(Local)").reset_index(drop=True)
    
    
    return monthly_area_df

"""fcr_d_2_directory = "../master-data/markets-data/FCR_D-2-2023.xlsx"

fcr_d_2_df_2023 = pd.read_excel(fcr_d_2_directory)

fcr_d_2_df_2023.columns

fcr_d_2_directory = "../master-data/markets-data/FCR_D-2-2023.xlsx"

filtered_d_2_df = preprocess_FCR(fcr_d_2_df_2023, start_month = 6, year = 2023, start_day = 14, end_month = 6, end_day = 30, start_hour = 0, end_hour = 23)

filtered_d_2_df

test = get_area_FCR_df(filtered_d_2_df, "NO4")
test"""

    
def create_FCR_dfs(fcr_d_1_path, fcr_d_2_path, start_month, year, start_day, end_month, end_day, start_hour, end_hour):
    fcr_d_1_df_2023 = pd.read_excel(fcr_d_1_path)
    fcr_d_2_df_2023 = pd.read_excel(fcr_d_2_path)
    
    filtered_d_1_df = preprocess_FCR(fcr_d_1_df_2023, start_month = start_month, year = year, start_day = start_day, end_month = end_month, end_day = end_day, start_hour = start_hour, end_hour = end_hour)
    filtered_d_2_df = preprocess_FCR(fcr_d_2_df_2023, start_month = start_month, year = year, start_day = start_day, end_month = end_month, end_day = end_day, start_hour = start_hour, end_hour = end_hour)

    fcr_d_1_dfs = []
    fcr_d_2_dfs = []
    
    areas = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']
    for area in areas:
        fcr_d_1_dfs.append(get_area_FCR_df(filtered_d_1_df, area))
        fcr_d_2_dfs.append(get_area_FCR_df(filtered_d_2_df, area))

        
    FCR_D_1_D_markets = []
    FCR_D_2_D_markets = []
    FCR_D_1_N_markets = []
    FCR_D_2_N_markets = []

    for df, area in zip(fcr_d_1_dfs, areas):
        FCR_D_1_D_markets.append(ReserveMarket(name = "FCR_D_D_1_" + area, direction = "up", area = area, response_time= 30, duration=15, min_volume=1, sleep_time= 60, activation_threshold= 49.9, price_data= df.drop(columns = ["FCR-N Price EUR/MW", "FCR-N Volume MW", "FCR-D Volume MW", "Area"]), volume_data= df.drop(columns = ["FCR-N Price EUR/MW", "FCR-N Volume MW", "FCR-D Price EUR/MW", "Area"])))
        FCR_D_1_N_markets.append(ReserveMarket(name = "FCR_N_D_1_" + area, direction = "both", area = area, response_time= 150, duration= 15, min_volume=1, sleep_time=60, activation_threshold= 50, price_data= df.drop(columns = ["FCR-D Price EUR/MW", "FCR-D Volume MW","FCR-N Volume MW", "Area"]), volume_data= df.drop(columns = ["FCR-D Price EUR/MW", "FCR-D Volume MW","FCR-N Price EUR/MW", "Area" ])))

    for df, area in zip(fcr_d_2_dfs, areas):
        FCR_D_2_D_markets.append(ReserveMarket(name = "FCR_D_D_2_" + area, direction = "up", area = area, response_time= 30, duration=15, min_volume=1, sleep_time= 60, activation_threshold= 49.9, price_data= df.drop(columns = ["FCR-N Price EUR/MW", "FCR-N Volume MW","FCR-D Volume MW", "Area"]), volume_data= df.drop(columns = ["FCR-N Price EUR/MW", "FCR-N Volume MW","FCR-D Price EUR/MW" , "Area"])))
        FCR_D_2_N_markets.append(ReserveMarket(name = "FCR_N_D_2_" + area, direction = "both", area = area, response_time= 150, duration= 15, min_volume=1, sleep_time=60, activation_threshold= 50, price_data= df.drop(columns = ["FCR-D Price EUR/MW", "FCR-D Volume MW","FCR-N Volume MW", "Area"]), volume_data= df.drop(columns = ["FCR-D Price EUR/MW", "FCR-D Volume MW","FCR-N Price EUR/MW" , "Area"])))

    return FCR_D_1_D_markets, FCR_D_2_D_markets, FCR_D_1_N_markets, FCR_D_2_N_markets

#___________________________________aFRR______________________________________

import os

def get_afrr_data(up_directory : str, down_directory : str):
    up_file_list = [file for file in os.listdir(up_directory) if file.endswith('.csv')]


    down_file_list = [file for file in os.listdir(down_directory) if file.endswith('.csv')]
    #merged_data = pd.DataFrame()

    up_data_files = []
    down_data_files = []

    for file in up_file_list:
        file_path = os.path.join(up_directory, file)
        data = pd.read_csv(file_path)
        up_data_files.append(data)

    for file in down_file_list:
        file_path = os.path.join(down_directory, file)
        data = pd.read_csv(file_path)
        down_data_files.append(data)
        
    up_df = pd.concat(up_data_files, ignore_index= True)
    down_df = pd.concat(down_data_files, ignore_index= True)
    return up_df, down_df

def preprocess_afrr(up_directory : str, down_directory : str, start_month : int, year : int, start_day : int, end_month : int, end_day : int, start_hour : int, end_hour: int):
    up_df, down_df = get_afrr_data(up_directory, down_directory)
    
    down_df = down_df.rename(columns = dict(zip(down_df.columns, ["Time(Local)", 
                                                              'aFRR Volume Down NO1','aFRR Price Down NO1', 
                                                              'aFRR Volume Down NO2','aFRR Price Down NO2',
                                                              'aFRR Volume Down NO3','aFRR Price Down NO3',
                                                              'aFRR Volume Down NO4','aFRR Price Down NO4',
                                                              'aFRR Volume Down NO5', 'aFRR Price Down NO5'])))
                
    up_df = up_df.rename(columns = dict(zip(up_df.columns, ["Time(Local)", 
                                                              'aFRR Volume Up NO1','aFRR Price Up NO1', 
                                                              'aFRR Volume Up NO2','aFRR Price Up NO2',
                                                              'aFRR Volume Up NO3','aFRR Price Up NO3',
                                                              'aFRR Volume Up NO4','aFRR Price Up NO4',
                                                              'aFRR Volume Up NO5', 'aFRR Price Up NO5'])))
    
    start_datetime = pd.Timestamp(year = year, month= start_month, day = start_day, hour = start_hour, tz = "Europe/Oslo")
    
        
    end_datetime = pd.Timestamp(year = year, month=end_month, day = end_day, hour = end_hour, tz = "Europe/Oslo")
    
    updated_dfs = []
    for df in [up_df, down_df]:
        df.sort_values(by= "Time(Local)", ignore_index= True, inplace= True)  
        
        df["Time(Local)"] = df["Time(Local)"].str.slice(0,16)
        
        df["Time(Local)"] = pd.to_datetime(df["Time(Local)"], format = '%d.%m.%Y %H:%M')
        
        df["Time(Local)"] = df["Time(Local)"].dt.tz_localize('Europe/Oslo', ambiguous='infer')
        
        filtered_df = df[(df["Time(Local)"] >= start_datetime) & (df["Time(Local)"] <= end_datetime)]
        updated_dfs.append(filtered_df)
        
    return updated_dfs[0], updated_dfs[1]

def get_area_afrr_dfs(df, area):
    # Filter by area
    
    removed_cols = df.columns[~df.columns.str.contains(area)]
    removed_cols = removed_cols[1:]
    area_df = df.drop(columns = removed_cols)
    # Sort by "Time(Local)" column
    area_df = area_df.sort_values(by="Time(Local)").reset_index(drop=True)
    return area_df

def create_afrr_dfs(up_directory, down_directory, year, start_month, start_day, start_hour, end_month, end_day, end_hour):
    afrr_area_up_dfs = []
    afrr_area_down_dfs = []
    areas = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']
    up_df, down_df = preprocess_afrr(up_directory = up_directory, down_directory = down_directory, year = year, start_month= start_month, end_month = end_month, start_day = start_day, end_day = end_day, start_hour= start_hour, end_hour= end_hour)
    for area in areas:
        afrr_area_up_dfs.append(get_area_afrr_dfs(up_df, area))
        afrr_area_down_dfs.append(get_area_afrr_dfs(down_df, area))
        
    aFRR_up_markets = []
    aFRR_down_markets = []
    for up_df, down_df, area in zip(afrr_area_up_dfs, afrr_area_down_dfs, areas):
        # bør kanskje legge inn et skille her mellom opp og ned?? Også fjerne activation threshold
        aFRR_up_markets.append(ReserveMarket(name = "aFRR up_" + area, direction = "up", area = area, response_time = 300, duration = 60, min_volume=1, sleep_time=60, activation_threshold=49.9, price_data = up_df.drop(columns = ["aFRR Volume Up " + area]), volume_data = up_df.drop(columns = ["aFRR Price Up " + area])))
        aFRR_down_markets.append(ReserveMarket(name = "aFRR down_" + area, direction = "down", area = area, response_time = 300, duration = 60, min_volume=1, sleep_time=60, activation_threshold=49.9, price_data = down_df.drop(columns = ["aFRR Volume Down " + area]), volume_data = down_df.drop(columns = ["aFRR Price Down " + area])))
    return aFRR_up_markets, aFRR_down_markets

"""afrr_up_directory = '../master-data/markets-data/aFFR/up_2023'
afrr_down_directory = '../master-data/markets-data/aFFR/down_2023'
aFRR_up_markets, aFRR_down_markets = create_afrr_dfs(up_directory = afrr_up_directory, down_directory = afrr_down_directory, year = tf.year, start_month = tf.start_month, start_day = tf.start_day, start_hour = tf.start_hour, end_month = tf.end_month, end_day = tf.end_day, end_hour = tf.end_hour)
"""

#________________________________RKOM______________________________________________

def initialize_rkom_df(rkom_22_path : str, rkom_23_path : str):
    rkom_2022_df = pd.read_excel(rkom_22_path)
    rkom_2023_df = pd.read_excel(rkom_23_path)
    rkom_dfs = [rkom_2022_df, rkom_2023_df]
    # remove all rows where hour is between 2-5 and between 7-24
    updated_dfs = []
    for df in rkom_dfs:
        rkom_df = df[~df['Hour'].isin([2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])]
        #change hour 1 to 1-5 and change hour 6 to 6-24
        for hour in rkom_df["Hour"]:
            if hour == 1:
                rkom_df["Hour"] = rkom_df["Hour"].replace(1, "1-5")
            elif hour == 6:
                rkom_df["Hour"] = rkom_df["Hour"].replace(6, "6-24")    
        updated_dfs.append(rkom_df)
    return updated_dfs[0], updated_dfs[1]



def get_hour_val_area_df(df, area : str, month, day, hour):
    year = df["Year"].iloc[0]
    #remove all rows where the are is equal to nan
    area_df = df.drop(df["Areas"][df["Areas"].isna()].index)

    #remove all rows where the chosen area is not present
    area_df = area_df.drop(area_df["Areas"].loc[(area_df["Areas"].str.contains(area) == False)].index)        
    #area_df = df.drop(df["Areas"][df["Areas"].str.contains(area) == False].index)        

    #Sort by week and then sort by hour within each week    
    area_df = area_df.sort_values(by=["Week", "Hour"])
    time_of_day = '1-5' if hour <= 5 else '6-24'
    date = datetime(year, month, day)
    week_num = date.isocalendar()[1]
    
    if len(area_df.loc[(area_df["Week"] == week_num)]) > 4:
        area_df = area_df.drop(area_df["Areas"][area_df["Areas"].str.contains("NO1,NO2,NO3,NO4,NO5")].index)
        area_df = area_df.fillna(0)
    else:
        area_df = area_df.fillna(0)
    
    area_df = area_df.loc[area_df["Week"] == week_num].reset_index(drop=True)
    return area_df.loc[(area_df["Hour"] == time_of_day)]

def create_standardized_RKOM_df(rkom_22_path : str, rkom_23_path : str, year, area, start_month, start_day, start_hour, end_month, end_day, end_hour):
    if year == 2022:
        df, _ = initialize_rkom_df(rkom_22_path, rkom_23_path)
    elif year == 2023:
        _, df = initialize_rkom_df(rkom_22_path, rkom_23_path)
    
    date_horizon =  pd.date_range(start=pd.Timestamp(year= year, month= start_month, day = start_day, hour = start_hour), 
                            end= pd.Timestamp(year = year, month = end_month, day = end_day, hour = end_hour), freq="H", tz = "Europe/Oslo")
    std_df = pd.DataFrame(np.zeros((len(date_horizon), 9)), columns= ["Time(Local)", "RKOM-H Price up", "RKOM-H Volume up", "RKOM-B Price up", "RKOM-B Volume up", "RKOM-H Price down", "RKOM-H Volume down", "RKOM-B Price down", "RKOM-B Volume down"])
    std_df["Time(Local)"] = date_horizon
    #print(date_horizon)
    
    NOK_EUR = 0.085
    for date in std_df["Time(Local)"]:
        month = date.month
        day = date.day
        hour = date.hour
        
        hour_val = get_hour_val_area_df(df, area, month, day, hour) 
        #print(hour_val)
        if date.weekday() < 5:
            std_df["RKOM-H Price up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Price Weekday"].iloc[0] * NOK_EUR # 0.085 for NOK to EUR
            std_df["RKOM-H Volume up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Volume Weekday"].iloc[0]
            std_df["RKOM-B Price up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Price Weekday"].iloc[0] * NOK_EUR
            std_df["RKOM-B Volume up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Volume Weekday"].iloc[0]
            std_df["RKOM-H Price down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Price Weekday"].iloc[1] * NOK_EUR
            std_df["RKOM-H Volume down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Volume Weekday"].iloc[1]
            std_df["RKOM-B Price down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Price Weekday"].iloc[1] * NOK_EUR
            std_df["RKOM-B Volume down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Volume Weekday"].iloc[1]
        else:
            std_df["RKOM-H Price up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Price Weekend"].iloc[0] * NOK_EUR
            std_df["RKOM-H Volume up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Volume Weekend"].iloc[0]
            std_df["RKOM-B Price up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Price Weekend"].iloc[0] * NOK_EUR
            std_df["RKOM-B Volume up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Volume Weekend"].iloc[0]
            std_df["RKOM-H Price down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Price Weekend"].iloc[1] * NOK_EUR
            std_df["RKOM-H Volume down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Volume Weekend"].iloc[1]
            std_df["RKOM-B Price down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Price Weekend"].iloc[1] * NOK_EUR
            std_df["RKOM-B Volume down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Volume Weekend"].iloc[1]
    
    return std_df

def create_RKOM_markets(rkom_22_path : str, rkom_23_path : str, year, start_month, start_day, start_hour, end_month, end_day, end_hour):
    rkom_dfs = []
    for area in areas:
        rkom_dfs.append(create_standardized_RKOM_df(rkom_22_path, rkom_23_path, year = year, area= area, start_month= start_month, start_day= start_day, start_hour=start_hour, end_month=end_month, end_day= end_day, end_hour= end_hour))
        

    RKOM_H_up_markets = []
    RKOM_H_down_markets = []
    RKOM_B_up_markets = []
    RKOM_B_down_markets = []

    for df, area in zip(rkom_dfs, areas):
        
        RKOM_H_up_markets.append(ReserveMarket(name = "RKOM_H_up_" + area, direction = "up", area = area, response_time = 300, duration = 60*4, min_volume = 5 if area == "NO1" or area == "NO3" else 10, sleep_time= 60, capacity_market = True, price_data= df.drop(columns = ["RKOM-H Volume up", "RKOM-H Volume down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-B Price up", "RKOM-B Price down", "RKOM-H Price down"]), volume_data= df.drop(columns = ["RKOM-H Price up", "RKOM-H Price down", "RKOM-B Price up", "RKOM-B Price down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-H Volume down"])))

        RKOM_H_down_markets.append(ReserveMarket(name = "RKOM_H_down_" + area, direction = "down",area = area, response_time = 300, duration = 60*4, min_volume = 5 if area == "NO1" or area == "NO3" else 10, sleep_time= 60, capacity_market = True, price_data= df.drop(columns = ["RKOM-H Volume up", "RKOM-H Volume down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-B Price up", "RKOM-B Price down", "RKOM-H Price up"]), volume_data= df.drop(columns = ["RKOM-H Price up", "RKOM-H Price down", "RKOM-B Price up", "RKOM-B Price down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-H Volume up"])))

        RKOM_B_up_markets.append(ReserveMarket(name = "RKOM_B_up_" + area, direction = "up",area = area, response_time = 300, duration = 60, min_volume = 5 if area == "NO1" or area == "NO3" else 10, sleep_time= 60*8, capacity_market = True, price_data= df.drop(columns = ["RKOM-H Volume up", "RKOM-H Volume down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-H Price up", "RKOM-H Price down", "RKOM-B Price down"]), volume_data= df.drop(columns = ["RKOM-H Price up", "RKOM-H Price down", "RKOM-B Price up", "RKOM-B Price down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-H Volume down"])))

        RKOM_B_down_markets.append(ReserveMarket(name = "RKOM_B_down_" + area, direction = "down", area = area,response_time = 300, duration = 60, min_volume = 5 if area == "NO1" or area == "NO3" else 10, sleep_time = 60*8, capacity_market = True, price_data= df.drop(columns = ["RKOM-H Volume up", "RKOM-H Volume down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-H Price up", "RKOM-H Price down", "RKOM-B Price up"]), volume_data= df.drop(columns = ["RKOM-H Price up", "RKOM-H Price down", "RKOM-B Price up", "RKOM-B Price down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-H Volume up"])))
        
    return RKOM_H_up_markets, RKOM_H_down_markets, RKOM_B_up_markets, RKOM_B_down_markets

# Stig Ødegaard Ottesen mente det ikke var noe problem å bruke 1 MW for RK
#________________________________________RKOM SESONG_____________________________________________
#RKOM_sesong = ReserveMarket(name = "RKOM_sesong", response_time=300, duration = 60, min_volume=10,sleep_time=0,activation_threshold=0, capacity_market= True, opening_date= datetime.datetime.strptime("2022-W44" + '-1', "%Y-W%W-%w"), end_date= datetime.datetime.strptime("2022-W17" + '-1', "%Y-W%W-%w"))

#_________________________________________________________________________________________________
def get_market_list(tf : Inputs.GlobalVariables, spot_path : str,  fcr_d_1_path : str,  fcr_d_2_path : str, afrr_up_directory : str, afrr_down_directory : str, rk_price_down_path : str, rk_price_up_path : str, rk_volume_down_path: str, rk_volume_up_path : str, rkom_22_path : str, rkom_23_path : str): 
    """Function to use all the functions defined in this file to create a list of all the markets that are to be used in the optimization problem.

    Args:
        tf (Inputs.GlobalVariables): the wanted timeframe for the optimization problem
        spot_path (str): path to the spot data
        fcr_d_1_path (str): path to the FCR-D1 dataset
        fcr_d_2_path (str): path to the FCR-D2 dataset
        afrr_up_directory (str): path to the aFRR up datasets
        afrr_down_directory (str): path to the aFRR down datasets
        rk_price_down_path (str): path to the RK price down dataset
        rk_price_up_path (str): path to the RK price up dataset
        rk_volume_down_path (str): path to the RK volume down dataset
        rk_volume_up_path (str): path to the RK volume up dataset
        rkom_22_path (str): path to the RKOM 2022 dataset
        rkom_23_path (str): path to the RKOM 2023 dataset

    Returns:
        list(ReserveMarket): list of all the possible  markets
    """
    FFR_markets = create_FFR_markets(start_year= tf.year, start_month= tf.start_month, start_day = tf.start_day, end_year = tf.year, end_month = tf.end_month, end_day = tf.end_day, start_hour = tf.start_hour, end_hour = tf.end_hour)
    FCR_D_1_D_markets, FCR_D_1_N_markets , FCR_D_2_N_markets , FCR_D_2_D_markets = create_FCR_dfs(fcr_d_1_path = fcr_d_1_path,  fcr_d_2_path = fcr_d_2_path, start_month = tf.start_month, year = tf.year, start_day = tf.start_day, end_month = tf.end_month, end_day = tf.end_day, start_hour = tf.start_hour, end_hour = tf.end_hour)
    aFRR_up_markets, aFRR_down_markets = create_afrr_dfs(up_df = afrr_up_directory, down_df = afrr_down_directory, year = tf.year, start_month = tf.start_month, start_day = tf.start_day, start_hour = tf.start_hour, end_month = tf.end_month, end_day = tf.end_day, end_hour = tf.end_hour)
    RK_down_markets , RK_up_markets = create_rk_markets(spot_path= spot_path, price_down_path = rk_price_down_path, price_up_path = rk_price_up_path, volume_down_path= rk_volume_down_path, volume_up_path = rk_volume_up_path, start_month = tf.start_month, year = tf.year, start_day = tf.start_day, start_hour = tf.start_hour, end_hour = tf.end_hour, end_month = tf.end_month, end_day = tf.end_day)
    RKOM_B_down_markets, RKOM_B_up_markets, RKOM_H_down_markets, RKOM_H_up_markets = create_RKOM_markets(rkom_22_path = rkom_22_path, rkom_23_path = rkom_23_path, year = tf.year, start_month = tf.start_month, start_day = tf.start_day, start_hour = tf.start_hour, end_month = tf.end_month, end_day = tf.end_day, end_hour = tf.end_hour)
    all_market_list = FFR_markets + FCR_D_1_D_markets + FCR_D_1_N_markets + FCR_D_2_N_markets + FCR_D_2_D_markets + aFRR_up_markets + aFRR_down_markets + RK_down_markets + RK_up_markets + RKOM_B_down_markets + RKOM_B_up_markets + RKOM_H_down_markets + RKOM_H_up_markets

    return all_market_list
    



#all_market_list = [FFR_flex, FFR_profile, FCR_D_1_D, FCR_D_1_N, FCR_D_2_D, FCR_D_2_N, aFRR_up, aFRR_down, RK_up, RK_down, RKOM_H_up, RKOM_H_down, RKOM_B_up, RKOM_B_down]

#[market.name for market in all_market_list]