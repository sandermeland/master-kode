from dataclasses import dataclass
import pandas as pd
import numpy as np
import datetime
import random as rand
from code_map import utils, timeframes

rand.seed(1337)

@dataclass
class PowerMeter:
    meter_id: str # metering_point_id
    response_time: int  # seconds
    up_flex_volume: pd.DataFrame  # dataframe of available flex volume to be regulated up each hour [MW]
    down_flex_volume: pd.DataFrame  # dataframe of available flex volume to be regulated down each hour [MW]
    direction: str  # 'up', 'down', or 'both'
    sleep_time: int  #  minutes the meter has to pause between each activation
    consumption_data: pd.DataFrame # The actual consumption data for the meter
    area: str # area of the metering point : NO1, NO2, NO3, NO4, NO5

        
def preprocess_consumption_df(df, tf : timeframes.TimeFrame):
    """ Function to preprocess the consumption data for a given time frame

    Args:
        df (pd.DataFrame): The unprocessed consumption data
        tf (timeframes.TimeFrame): The wanted timeframe
    
    Returns:
        df (pd.DataFrame): The processed consumption data for the given timeframe
    """
    df["category"] = pd.Series(np.zeros(len(df)), dtype = "str")
    start_date = pd.Timestamp(tf.year, tf.start_month, tf.start_day, tf.start_hour).tz_localize('Europe/Oslo')
    end_date = pd.Timestamp(tf.year, tf.end_month, tf.end_day, tf.end_hour).tz_localize('Europe/Oslo')
   
    df["start_time_local"] = pd.to_datetime(df["start_time_local"])
 
    df["start_time_local"] = df["start_time_local"].dt.tz_convert('Europe/Oslo')
   
    df = df.loc[(df["start_time_local"] >= start_date) & (df["start_time_local"] <= end_date)]
   
    hours_in_timehorizone = (end_date - start_date).days*24 + (end_date - start_date).seconds/3600
    
    count_hours_df = df.groupby("metering_point_id")["value"].agg(["count"])
    missing_hours_df = count_hours_df.loc[(count_hours_df["count"] < hours_in_timehorizone)]
    new_meter_ids = missing_hours_df.index.tolist()
    df = df[(~df["metering_point_id"].isin(new_meter_ids))]
    
    df.rename(columns={'start_time_local':'Time(Local)'}, inplace=True)
    df["value"] = df["value"] * 0.001 # convert from KWh to MWh
    return df


def combine_category_dfs(list_of_paths : list):
    dfs = []
    for path in list_of_paths:
        df = pd.read_csv(path)
        df["category"] = path[35:-4]
        cols_to_drop = [col for col in df.columns if col.startswith("Meter")]
        # Drop these columns
        df.drop(columns=cols_to_drop, inplace=True)        
        dfs.append(df)
    return pd.concat(dfs, ignore_index = True)
        
def create_meter_objects(consumption_data : pd.DataFrame ,tf : timeframes.TimeFrame, reference_tf : timeframes.TimeFrame, category_path_list : list ):
    """
    Creates the meter objects from the consumption data. 
    The flex volume is calculated as the difference between the min/max value for the same hour and day of the week in the reference timeframe and the consumption data for the timeframe.
    The response time is a random number between 0 and 300 seconds.
    The sleep time is a random number between 0 and 30 minutes.
    The direction is either up, down or both. 80 percent are chosen to be both while the last 20 percent are randomly chosen to be either up, both or down.
    
    Args:
        consumption_data (pd.DataFrame): The consumption data
        tf (Inputs.timeframes.TimeFrame): The wanted timeframe
        reference_tf (timeframes.TimeFrame): The reference timeframe for the flex volume to find min/max values
        
    Returns:
        dict: a dictionary of the power meters
    """
    power_meters = {}
    updated_df = preprocess_consumption_df(consumption_data, tf).copy()
    monthly_df = preprocess_consumption_df(consumption_data, reference_tf).copy()
    monthly_df['day_of_week'] = monthly_df['Time(Local)'].dt.day_name().astype('category')
    monthly_df['hour'] = monthly_df['Time(Local)'].dt.hour


    # Compute min and max values
    grouped = monthly_df.groupby(['metering_point_id', 'day_of_week', 'hour'], observed=True)
    aggregates = grouped['value'].agg(['min', 'max']).reset_index()

    # Create dictionary for fast lookup
    lookup_dict = {(row['metering_point_id'], row['day_of_week'], row['hour']): (row['min'], row['max']) for index, row in aggregates.iterrows()}

    category_df = combine_category_dfs(category_path_list)

    for counter, meter_id in enumerate(updated_df["metering_point_id"].unique()):
        # the meter_id has to have values for all hours both in the timeframe and the reference timeframe
        if meter_id in monthly_df["metering_point_id"].unique():
            
            meter_values = updated_df.loc[(updated_df["metering_point_id"] == meter_id)]
            meter_values = meter_values.drop(columns = "metering_point_id")
            
            # calculate response time based on the categories
            # response time = mean of the flex_volume divided by the max consumption in the dataset for that meter times a value

            if meter_id in category_df["Identification"].unique():
                if category_df["category"].loc[category_df["Identification"] == meter_id].iloc[0] == "ev_meters":
                    response_time = 1
                elif category_df["category"].loc[category_df["Identification"] == meter_id].iloc[0] == "harktech_meters":
                    response_time = 5
                else:
                    response_time = rand.random()*300
            else:
                response_time = rand.random()*300
            # Set direction
            directions = ["up", "down", "both"]
            direction_index = rand.randint(0,2) if counter % 5 == 0 else 2
            # Set area
            area = updated_df['area'].loc[(updated_df["metering_point_id"] == meter_id)].iloc[0]
            
            if direction_index == 0: # up
                up_flex_volume = meter_values.copy()
                up_flex_volume['weekday'] = up_flex_volume['Time(Local)'].dt.strftime('%A')
                up_flex_volume['hour'] = up_flex_volume['Time(Local)'].dt.hour
                up_flex_volume.set_index(['weekday', 'hour'], inplace=True)

                # Using .map() to subtract the min values from 'value' column based on 'weekday' and 'hour'
                up_flex_volume['value'] = up_flex_volume['value'] - up_flex_volume.index.map(lambda x: lookup_dict.get((meter_id, x[0], x[1]), (0,0))[0])

                up_flex_volume.reset_index(inplace=True)
                meter = PowerMeter(meter_id = meter_id, response_time = response_time, up_flex_volume= up_flex_volume , down_flex_volume = [], direction = directions[direction_index], sleep_time = rand.random()*30, consumption_data = meter_values, area = area )
            
            elif direction_index == 1:
                down_flex_volume = meter_values.copy()
                down_flex_volume['weekday'] = down_flex_volume['Time(Local)'].dt.strftime('%A')
                down_flex_volume['hour'] = down_flex_volume['Time(Local)'].dt.hour
                down_flex_volume.set_index(['weekday', 'hour'], inplace=True)

                # Using .map() to subtract the max values from 'value' column based on 'weekday' and 'hour'
                down_flex_volume['value'] = down_flex_volume['value'] - down_flex_volume.index.map(lambda x: lookup_dict.get((meter_id, x[0], x[1]), (0,0))[0])

                down_flex_volume.reset_index(inplace=True)
                meter = PowerMeter(meter_id = meter_id, response_time = response_time, up_flex_volume= [] , down_flex_volume = down_flex_volume, direction = directions[direction_index], sleep_time = rand.random()*30, consumption_data = meter_values, area = area )
            else:
                up_flex_volume = meter_values.copy()
                up_flex_volume['weekday'] = up_flex_volume['Time(Local)'].dt.strftime('%A')
                up_flex_volume['hour'] = up_flex_volume['Time(Local)'].dt.hour
                up_flex_volume.set_index(['weekday', 'hour'], inplace=True)

                # Using .map() to subtract the min values from 'value' column based on 'weekday' and 'hour'
                up_flex_volume['value'] = up_flex_volume['value'] - up_flex_volume.index.map(lambda x: lookup_dict.get((meter_id, x[0], x[1]), (0,0))[0])

                up_flex_volume.reset_index(inplace=True)
                down_flex_volume = meter_values.copy()
                down_flex_volume['weekday'] = down_flex_volume['Time(Local)'].dt.strftime('%A')
                down_flex_volume['hour'] = down_flex_volume['Time(Local)'].dt.hour
                down_flex_volume.set_index(['weekday', 'hour'], inplace=True)

                # Using .map() to subtract the max values from 'value' column based on 'weekday' and 'hour'
                down_flex_volume['value'] = down_flex_volume['value'] - down_flex_volume.index.map(lambda x: lookup_dict.get((meter_id, x[0], x[1]), (0,0))[0])

                down_flex_volume.reset_index(inplace=True)
                meter = PowerMeter(meter_id = meter_id, response_time = response_time, up_flex_volume= up_flex_volume , down_flex_volume = down_flex_volume, direction = directions[direction_index], sleep_time = rand.random()*30, consumption_data = meter_values, area = area )

            power_meters[meter_id] = meter
        else:
            continue
    return power_meters



"""

Når det kommer til response time hadde ikke kjartan noen innspill. Det kan være vanskelig å gjøre noe med kategorisering da vi ikke har oversikt over dette.

Up/Down flex volume should be dependent of the same hour from previous same days of the week. Meaning that if we are looking at a monday at 12:00, we should look at all mondays at 12:00 and take the min/max of the values. This should be done for all hours in the timehorizone.

old version : 
up_flex_volume = meter_values.copy()
up_flex_volume["value"] = up_flex_volume["value"] - min(up_flex_volume["value"])

down_flex_volume = meter_values.copy()
down_flex_volume["value"] = max(down_flex_volume["value"]) - down_flex_volume["value"] 

    The new method is way to slow and i have to speed it up. 
    One alternative is to make a df for each day of the week and hour which has the min/max values

updated_old_version (which gave a lot of warnings due to trying to set values on a copy of a slice from a DataFrame)) : 
down_flex_volume["value"].loc[down_flex_volume["Time(Local)"] == hour] =  lookup_dict[(meter_id, hour.strftime('%A'), hour.hour)][1] - down_flex_volume["value"].loc[down_flex_volume["Time(Local)"] == hour]
(same for up_flex_volume)

"""
