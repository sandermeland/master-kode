import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar 
from datetime import datetime
import pytz
import openpyxl
from code_map import Utils

## Tror det mest sannsynlig ikke er nødvendig med activation_threshold. Kan da også smekke sammen de 2 ffr markedene. Ankepunktet er eventuelt 
## Bør jeg ha med opp og ned?
## For FCR bør jeg nok ha med D-2 og D-1
## Aktiveringstid bør ikke være så farlig ettersom det er aggregert. 
### Ettersom en regulering for x antall laster skjer parallellet vil det gå mye raskere
#### Gjennomsnittlig responstid delt på antall laster??

#market_list = [FFR_flex, FFR_profile, FCR_D_D_1, , FCR_N, aFRR_up, RK, RKOM_sesong, RKOM_uke]
    
#print(market_list)

"""
Tanker 19/10
Bør i utgangspunktet ha data fra en tidlig måned i 2023 ettersom aFRR ble endret i slutten av 2022 og at FCR-D opp var ikke oppe og gikk før nylig
Fra nåværende samlet dataframe for markedene er det 28 markeder. Her mangler RKOM sesong. Dette kan dog deles på to da det er både volum og pris. FFR har bare pris, dvs 12 markeder totalt
"""

class ReserveMarket:
    """
    response_time : how fast the market needs the power meter to react (seconds)
    duration : how fast the market needs the power meter to be activated (minutes)
    min_volume : the minimum volume needed in the market (MW)
    sleep_time : maximum allowed sleep time (minutes)
    activation_threshold : threshold where the market is 
    """
    def __init__(self, name, response_time, duration, min_volume, volume_data, price_data, direction, area, sleep_time = 60, activation_threshold = 0,  capacity_market = True):
        self.response_time = response_time
        self.duration = duration
        self.min_volume = min_volume
        self.sleep_time = sleep_time
        self.activation_threshold = activation_threshold
       # self.available_hours = available_hours
        self.capacity_market = capacity_market
        self.name = name
        #self.opening_date = opening_date
        #self.end_date = end_date
        self.volume_data = volume_data
        self.price_data = price_data
        self.direction = direction
        self.area = area

#________________________________Global variables____________________________________
version_variables = Utils.one_day
year = version_variables.year
start_month = version_variables.start_month
end_month = version_variables.end_month
start_day = version_variables.start_day
end_day = version_variables.end_day
start_hour = version_variables.start_hour
end_hour = version_variables.end_hour
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

# opening_date= datetime.datetime(2022, 5, 27), end_date= datetime.datetime(2022,9,3)
ffr_data = get_FFR_df(start_year = year, start_month = start_month, start_day = start_day, start_hour = start_hour, end_year = year, end_month = end_month, end_day = end_day, end_hour = end_hour)

#print(ffr_data)


FFR_profile = ReserveMarket(name = "FFR_profile", direction = "up", area = "all", response_time= 1.3, duration= 0.5, min_volume= 1, sleep_time=15, activation_threshold= 49.7, capacity_market= True, price_data= ffr_data.drop(columns= ["FFR-Flex Volume", "FFR-Flex Price [EUR/MW]", "FFR-Profil Volume"]), volume_data= ffr_data.drop(columns= ["FFR-Flex Price [EUR/MW]", "FFR-Flex Volume", "FFR-Profil Price [EUR/MW]"]))

FFR_flex = ReserveMarket(name = "FFR_flex", direction = "up", area = "all", response_time= 1.3, duration= 0.5, min_volume= 5, sleep_time=15, activation_threshold= 49.7, price_data= ffr_data.drop(columns= ["FFR-Flex Volume", "FFR-Profil Price [EUR/MW]", "FFR-Profil Volume"]), volume_data= ffr_data.drop(columns= ["FFR-Flex Price [EUR/MW]", "FFR-Profil Volume", "FFR-Profil Price [EUR/MW]"]))

FFR_markets = [FFR_flex, FFR_profile]

#_____________________________________FCR_____________________________________

#fcr_d_1_df_2022 = pd.read_excel("/Users/sandermeland/Documents/Jobb/Volte/master-kode/markets/markets-data/new_fcrd1.xlsx")
#fcr_d_2_df_2022 = pd.read_excel("/Users/sandermeland/Documents/Jobb/Volte/master-kode/markets/markets-data/new_fcrd2.xlsx")
fcr_d_1_df_2023 = pd.read_excel("../master-data/markets-data/FCR_D-1-2023.xlsx", engine = 'openpyxl')
fcr_d_2_df_2023 = pd.read_excel("../master-data/markets-data/FCR_D-2-2023.xlsx")

def preprocess_FCR(df: pd.DataFrame, area : str, start_month : int, year : int, start_day : int, end_month : int, end_day : int, start_hour : int, end_hour : int):
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
    
    # Filter by area
    monthly_area_df = filtered_df[filtered_df["Area"] == area]
    
    # Sort by "Time(Local)" column
    monthly_area_df = monthly_area_df.sort_values(by="Time(Local)").reset_index(drop=True)
    
    return monthly_area_df
    
fcr_d_1_dfs = []
fcr_d_2_dfs = []
for area in areas:
    fcr_d_1_dfs.append(preprocess_FCR(fcr_d_1_df_2023, area = area, start_month = start_month, year = year, start_day = start_day, end_month = end_month, end_day = end_day, start_hour = start_hour, end_hour = end_hour))
    fcr_d_2_dfs.append(preprocess_FCR(fcr_d_2_df_2023, area = area, start_month = start_month, year = year, start_day = start_day, end_month = end_month, end_day = end_day, start_hour = start_hour, end_hour = end_hour))

FCR_D_1_D_markets = []
FCR_D_2_D_markets = []
FCR_D_1_N_markets = []
FCR_D_2_N_markets = []

for df, area in zip(fcr_d_1_dfs, areas):
    FCR_D_1_D_markets.append(ReserveMarket(name = "FCR_D_D_1_" + area, direction = "up", area = area, response_time= 30, duration=15, min_volume=1, sleep_time= 60, activation_threshold= 49.9, price_data= df.drop(columns = ["FCR-N Price EUR/MW", "FCR-N Volume MW", "FCR-D Volume MW", "Area"]), volume_data= df.drop(columns = ["FCR-N Price EUR/MW", "FCR-N Volume MW", "FCR-D Price EUR/MW", "Area"])))
    FCR_D_1_N_markets.append(ReserveMarket(name = "FCR_N_D_1_" + area, direction = "both", area = area, response_time= 30, duration= 15, min_volume=1, sleep_time=60, activation_threshold= 50, price_data= df.drop(columns = ["FCR-D Price EUR/MW", "FCR-D Volume MW","FCR-N Volume MW", "Area"]), volume_data= df.drop(columns = ["FCR-D Price EUR/MW", "FCR-D Volume MW","FCR-N Price EUR/MW", "Area" ])))

for df, area in zip(fcr_d_2_dfs, areas):
    FCR_D_2_D_markets.append(ReserveMarket(name = "FCR_D_D_2_" + area, direction = "up", area = area, response_time= 30, duration=15, min_volume=1, sleep_time= 60, activation_threshold= 49.9, price_data= df.drop(columns = ["FCR-N Price EUR/MW", "FCR-N Volume MW","FCR-D Volume MW", "Area"]), volume_data= df.drop(columns = ["FCR-N Price EUR/MW", "FCR-N Volume MW","FCR-D Price EUR/MW" , "Area"])))
    FCR_D_2_N_markets.append(ReserveMarket(name = "FCR_N_D_2_" + area, direction = "both", area = area, response_time= 30, duration= 15, min_volume=1, sleep_time=60, activation_threshold= 50, price_data= df.drop(columns = ["FCR-D Price EUR/MW", "FCR-D Volume MW","FCR-N Volume MW", "Area"]), volume_data= df.drop(columns = ["FCR-D Price EUR/MW", "FCR-D Volume MW","FCR-N Price EUR/MW" , "Area"])))


#___________________________________aFRR______________________________________

import os

up_directory = '../master-data/markets-data/aFFR/up_2023'
up_file_list = [file for file in os.listdir(up_directory) if file.endswith('.csv')]

down_directory = '../master-data/markets-data/aFFR/down_2023'
down_file_list = [file for file in os.listdir(down_directory) if file.endswith('.csv')]
merged_data = pd.DataFrame()

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

def preprocess_afrr(up_df : pd.DataFrame, down_df : pd.DataFrame,  area : str, start_month : int, year : int, start_day : int, end_month : int, end_day : int, start_hour : int, end_hour: int):
    
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

        removed_cols = df.columns[~df.columns.str.contains(area)]
        removed_cols = removed_cols[1:]
        # Filter by area
        area_df = filtered_df.drop(columns = removed_cols)
        
        # Sort by "Time(Local)" column
        area_df = area_df.sort_values(by="Time(Local)").reset_index(drop=True)
        updated_dfs.append(area_df)
    return updated_dfs

afrr_area_dfs = []
for area in areas:
    afrr_area_dfs.append(preprocess_afrr(up_df, down_df, area, year = 2023, start_month= start_month, end_month = end_month, start_day = start_day, end_day = end_day, start_hour= start_hour, end_hour= end_hour))

aFRR_up_markets = []
aFRR_down_markets = []
for df, area in zip(afrr_area_dfs, areas):
    aFRR_up_markets.append(ReserveMarket(name = "aFRR up_" + area, direction = "up", area = area, response_time = 300, duration = 60, min_volume=1, sleep_time=60, activation_threshold=49.9, price_data = df[0].drop(columns = ["aFRR Volume Up " + area]), volume_data = df[0].drop(columns = ["aFRR Price Up " + area])))
    aFRR_down_markets.append(ReserveMarket(name = "aFRR down_" + area, direction = "down", area = area, response_time = 300, duration = 60, min_volume=1, sleep_time=60, activation_threshold=49.9, price_data = df[0].drop(columns = ["aFRR Volume Up " + area]), volume_data = df[0].drop(columns = ["aFRR Price Up " + area])))

#___________________________________RK_________________________________________

rk_price_down = pd.read_csv("../master-data/markets-data/RK/new_rk_price_down.csv")
rk_price_up = pd.read_csv("../master-data/markets-data/RK/new_rk_price_up.csv")
rk_volume_up = pd.read_csv("../master-data/markets-data/RK/new_rk_vol_up.csv")
rk_volume_down = pd.read_csv("../master-data/markets-data/RK/new_rk_vol_down.csv")

rk_price_down.drop(columns = ["currency"], inplace=True)
rk_price_up.drop(columns = ["currency"], inplace=True)

rk_dfs_dict = {"price_down" : rk_price_down,"price_up" : rk_price_up,"volume_up" : rk_volume_up,"volume_down" : rk_volume_down}


def preprocess_rk_dfs_dict(df_dict : dict, area : str, start_month : int, start_year : int, start_day : int, start_hour : int, end_hour : int, end_month : int, end_year : int, end_day : int):
    start_datetime = pd.Timestamp(year = start_year, month= start_month, day=start_day, hour= start_hour, tz = "Europe/Oslo") #Europe/Oslo    
    end_datetime = pd.Timestamp(year = end_year, month= end_month, day=end_day, hour= end_hour, tz = "Europe/Oslo")
    updated_df_dict = {}
    for name in df_dict.keys():
        df = df_dict[name].copy()
        if name == "volume_down":
            df["value"].loc[df["value"] != 0] = df["value"].loc[df["value"] != 0] * -1
        #df.drop(columns = ["end_time", "bi_created", "start_year" ], inplace = True)
        df["start_time"] = pd.to_datetime(df["start_time"], format="%Y-%m-%d %H:%M:%S")
        #df["start_time"] = df["start_time"].dt.tz_localize("UTC")
        df["start_time"] = df["start_time"].dt.tz_convert("Europe/Oslo")
        df.sort_values(by = ["start_time", "delivery_area"], inplace = True)
        df.rename(columns = {"start_time" : "Time(Local)"}, inplace = True)
        filtered_df = df[(df["Time(Local)"] >= start_datetime) & (df["Time(Local)"] <= end_datetime) & (df["delivery_area"] == area)]
        filtered_df.sort_values(by = ["Time(Local)"], inplace = True)
        updated_df_dict[name] = filtered_df
        
    return updated_df_dict

rk_dicts = []
for area in areas:
    rk_dicts.append(preprocess_rk_dfs_dict(rk_dfs_dict, area=area, start_year= year, end_year = year, start_month = start_month, end_month = end_month, start_day = start_day, end_day = end_day, start_hour = start_hour, end_hour = end_hour))

RK_up_markets = []
RK_down_markets = []

for rk_dict, area in zip(rk_dicts, areas):
    RK_up_markets.append(ReserveMarket(name = "RK_up_" + area, direction = "up", area = area, capacity_market= False, response_time=300, duration = 60, min_volume=10,sleep_time=0,activation_threshold=0, price_data= rk_dict["price_up"], volume_data= rk_dict["volume_up"]))
    RK_down_markets.append(ReserveMarket(name = "RK_down_" + area, direction = "down", area = area, response_time=300, capacity_market=False,  duration = 60, min_volume=10,sleep_time=0,activation_threshold=0, price_data= rk_dict["price_down"], volume_data= rk_dict["volume_down"]))



#________________________________RKOM______________________________________________

rkom_2022_df = pd.read_excel("../master-data/markets-data/RKOM.xlsx")
rkom_2023_df = pd.read_excel("../master-data/markets-data/Rkom-2023.xlsx")
rkom_dfs = [rkom_2022_df, rkom_2023_df]

def preprocess_rkom_df(df_list):
    # remove all rows where hour is between 2-5 and between 7-24
    updated_dfs = []
    for df in df_list:
        rkom_df = df[~df['Hour'].isin([2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])]
        #change hour 1 to 1-5 and change hour 6 to 6-24
        for hour in rkom_df["Hour"]:
            if hour == 1:
                rkom_df["Hour"] = rkom_df["Hour"].replace(1, "1-5")
            elif hour == 6:
                rkom_df["Hour"] = rkom_df["Hour"].replace(6, "6-24")    
        updated_dfs.append(rkom_df)
    return updated_dfs[0], updated_dfs[1]

rkom_22, rkom_23 = preprocess_rkom_df(rkom_dfs)

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


def create_standardized_RKOM_df(df_list, year, area, start_month, start_day, start_hour, end_month, end_day, end_hour):
    
    if year == 2022:
        df = df_list[0]
    else:
        df = df_list[1]
    
    date_horizon =  pd.date_range(start=pd.Timestamp(year= year, month= start_month, day = start_day, hour = start_hour), 
                            end= pd.Timestamp(year = year, month = end_month, day = end_day, hour = end_hour), freq="H", tz = "Europe/Oslo")
    std_df = pd.DataFrame(np.zeros((len(date_horizon), 9)), columns= ["Time(Local)", "RKOM-H Price up", "RKOM-H Volume up", "RKOM-B Price up", "RKOM-B Volume up", "RKOM-H Price down", "RKOM-H Volume down", "RKOM-B Price down", "RKOM-B Volume down"])
    std_df["Time(Local)"] = date_horizon
    #print(date_horizon)
    for date in std_df["Time(Local)"]:
        month = date.month
        day = date.day
        hour = date.hour
        
        hour_val = get_hour_val_area_df(df, area, month, day, hour) 
        #print(hour_val)
        if date.weekday() < 5:
            std_df["RKOM-H Price up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Price Weekday"].iloc[0] * 0.085
            std_df["RKOM-H Volume up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Volume Weekday"].iloc[0]
            std_df["RKOM-B Price up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Price Weekday"].iloc[0] * 0.085
            std_df["RKOM-B Volume up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Volume Weekday"].iloc[0]
            std_df["RKOM-H Price down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Price Weekday"].iloc[1] * 0.085
            std_df["RKOM-H Volume down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Volume Weekday"].iloc[1]
            std_df["RKOM-B Price down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Price Weekday"].iloc[1] * 0.085
            std_df["RKOM-B Volume down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Volume Weekday"].iloc[1]
        else:
            std_df["RKOM-H Price up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Price Weekend"].iloc[0] * 0.085
            std_df["RKOM-H Volume up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Volume Weekend"].iloc[0]
            std_df["RKOM-B Price up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Price Weekend"].iloc[0] * 0.085
            std_df["RKOM-B Volume up"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Volume Weekend"].iloc[0]
            std_df["RKOM-H Price down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Price Weekend"].iloc[1] * 0.085
            std_df["RKOM-H Volume down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-H Volume Weekend"].iloc[1]
            std_df["RKOM-B Price down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Price Weekend"].iloc[1] * 0.085
            std_df["RKOM-B Volume down"][(std_df["Time(Local)"] == date)] = hour_val["RKOM-B Volume Weekend"].iloc[1]
    
    return std_df

rkom_dfs = []
for area in areas:
    rkom_dfs.append(create_standardized_RKOM_df([rkom_22, rkom_23], year = year, area= area, start_month= start_month, start_day= start_day, start_hour=start_hour, end_month=end_month, end_day= end_day, end_hour= end_hour))
    

RKOM_H_up_markets = []
RKOM_H_down_markets = []
RKOM_B_up_markets = []
RKOM_B_down_markets = []

for df, area in zip(rkom_dfs, areas):
    RKOM_H_up_markets.append(ReserveMarket(name = "RKOM_H_up_" + area, direction = "up", area = area, response_time = 300, duration = 60*4, min_volume = 10, sleep_time= 60, capacity_market = True, price_data= df.drop(columns = ["RKOM-H Volume up", "RKOM-H Volume down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-B Price up", "RKOM-B Price down", "RKOM-H Price down"]), volume_data= df.drop(columns = ["RKOM-H Price up", "RKOM-H Price down", "RKOM-B Price up", "RKOM-B Price down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-H Volume down"])))

    RKOM_H_down_markets.append(ReserveMarket(name = "RKOM_H_down_" + area, direction = "down",area = area, response_time = 300, duration = 60*4, min_volume = 10, sleep_time= 60, capacity_market = True, price_data= df.drop(columns = ["RKOM-H Volume up", "RKOM-H Volume down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-B Price up", "RKOM-B Price down", "RKOM-H Price up"]), volume_data= df.drop(columns = ["RKOM-H Price up", "RKOM-H Price down", "RKOM-B Price up", "RKOM-B Price down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-H Volume up"])))

    RKOM_B_up_markets.append(ReserveMarket(name = "RKOM_B_up_" + area, direction = "up",area = area, response_time = 300, duration = 60, min_volume = 10, sleep_time= 60*8, capacity_market = True, price_data= df.drop(columns = ["RKOM-H Volume up", "RKOM-H Volume down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-H Price up", "RKOM-H Price down", "RKOM-B Price down"]), volume_data= df.drop(columns = ["RKOM-H Price up", "RKOM-H Price down", "RKOM-B Price up", "RKOM-B Price down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-H Volume down"])))

    RKOM_B_down_markets.append(ReserveMarket(name = "RKOM_B_down_" + area, direction = "down", area = area,response_time = 300, duration = 60, min_volume = 10, sleep_time = 60*8, capacity_market = True, price_data= df.drop(columns = ["RKOM-H Volume up", "RKOM-H Volume down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-H Price up", "RKOM-H Price down", "RKOM-B Price up"]), volume_data= df.drop(columns = ["RKOM-H Price up", "RKOM-H Price down", "RKOM-B Price up", "RKOM-B Price down", "RKOM-B Volume up", "RKOM-B Volume down", "RKOM-H Volume up"])))



#________________________________________RKOM SESONG_____________________________________________
#RKOM_sesong = ReserveMarket(name = "RKOM_sesong", response_time=300, duration = 60, min_volume=10,sleep_time=0,activation_threshold=0, capacity_market= True, opening_date= datetime.datetime.strptime("2022-W44" + '-1', "%Y-W%W-%w"), end_date= datetime.datetime.strptime("2022-W17" + '-1', "%Y-W%W-%w"))

#_________________________________________________________________________________________________
def get_market_list(tf : Utils.GlobalVariables):
    """Function to get all markets. Might be to stressfull to make this function. Will have to load all of the datasets in this function. It will be extremely long.

    Args:
        tf (Inputs.GlobalVariables): _description_
    """
    return None
    

#all_market_list = [FFR_flex, FFR_profile, FCR_D_1_D, FCR_D_1_N, FCR_D_2_D, FCR_D_2_N, aFRR_up, aFRR_down, RK_up, RK_down, RKOM_H_up, RKOM_H_down, RKOM_B_up, RKOM_B_down]
all_market_list = FFR_markets + FCR_D_1_D_markets + FCR_D_1_N_markets + FCR_D_2_N_markets + FCR_D_2_D_markets + aFRR_up_markets + aFRR_down_markets + RK_down_markets + RK_up_markets + RKOM_B_down_markets + RKOM_B_up_markets + RKOM_H_down_markets + RKOM_H_up_markets

[market.name for market in all_market_list]