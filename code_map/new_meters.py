from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random as rand
from code_map import Inputs

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

        



def preprocess_df(df, tf : Inputs.GlobalVariables):
    start_date = pd.Timestamp(tf.year, tf.start_month, tf.start_day, tf.start_hour).tz_localize('Europe/Oslo')
    end_date = pd.Timestamp(tf.year, tf.end_month, tf.end_day, tf.end_hour).tz_localize('Europe/Oslo')
    df["start_time_local"] = pd.to_datetime(df["start_time_local"])
    #print(df["start_time_local"])
    df["start_time_local"] = df["start_time_local"].dt.tz_convert('Europe/Oslo')
    #print(df["start_time_local"])
    df = df.loc[(df["start_time_local"] >= start_date) & (df["start_time_local"] <= end_date)]
   # print(df["start_time_local"])
    hours_in_timehorizone = (end_date - start_date).days*24 + (end_date - start_date).seconds/3600
    #print(hours_in_timehorizone)
    # for eac metering_point_id, check if there is data for the whole timehorizone
    count_hours_df = df.groupby("metering_point_id")["value"].agg(["count"])
   # print(count_hours_df)
    missing_hours_df = count_hours_df.loc[(count_hours_df["count"] < hours_in_timehorizone)]
    new_meter_ids = missing_hours_df.index.tolist()
    df = df[(~df["metering_point_id"].isin(new_meter_ids))]
    #print(df)
    
    df.rename(columns={'start_time_local':'Time(Local)'}, inplace=True)
    #df.drop(columns = ["end_time_local"], inplace = True)
    df["value"] = df["value"] * 0.001 # convert from KWh to MWh
    return df





def find_extreme_val_for_hour_in_month(df : pd.DataFrame, hour : pd.Timestamp, min : bool):
    """
    Find all values in the DataFrame corresponding to the same hour and day of the week as 'hour'.
    Return the lowest of these values.
    
    Args:
    df (pd.DataFrame): DataFrame with columns 'Time' and 'Values'
    hour (str or pd.Timestamp): The specific hour to match in format 'YYYY-MM-DD-HH:MM'
    
    Returns:
    float: The lowest value found for the same hour and day of week.
    """
    
    # Extract the hour and day of the week from the given 'hour'
    hour_to_match = hour.hour
    weekday_to_match = hour.dayofweek
    
    # Filter the DataFrame for entries that match the hour and day of the week
    mask = (df['Time(Local)'].dt.hour == hour_to_match) & (df['Time(Local)'].dt.dayofweek == weekday_to_match)
    matching_values = df.loc[mask, 'value']
    
    # Return the lowest value among the matching entries
    return matching_values.min() if min else matching_values.max()

def adjust_values_by_weekday_min(df_24h, reference_df):
    """
    Adjusts values in a 24-hour DataFrame based on the minimum values for the same hour and weekday in a reference DataFrame.

    Args:
        df_24h (pd.DataFrame): DataFrame with 24-hour values to adjust.
        reference_df (pd.DataFrame): Reference DataFrame to calculate minimum values from.

    Returns:
        pd.DataFrame: Adjusted DataFrame with values reflecting the difference from the minimum.
    """

    # Ensure the 'Time' column in both DataFrames is in datetime format
    df_24h['Time'] = pd.to_datetime(df_24h['Time'])
    reference_df['Time'] = pd.to_datetime(reference_df['Time'])

    # Add helper columns for merging
    reference_df['weekday'] = reference_df['Time'].dt.dayofweek
    reference_df['hour'] = reference_df['Time'].dt.hour

    # Find the minimum value for each hour and weekday combination
    min_values = (
        reference_df.groupby(['weekday', 'hour'])['Values']
        .min()
        .reset_index()
        .rename(columns={'Values': 'min_value'})
    )

    # Add the same helper columns to the 24-hour DataFrame for merging
    df_24h['weekday'] = df_24h['Time'].dt.weekday
    df_24h['hour'] = df_24h['Time'].dt.hour

    # Merge the minimum values into the 24-hour DataFrame
    df_with_mins = pd.merge(df_24h, min_values, on=['weekday', 'hour'], how='left')

    # Calculate the difference
    df_with_mins['Value_Difference'] = df_with_mins['Values'] - df_with_mins['min_value']

    # Drop unnecessary columns
    df_with_mins.drop(['weekday', 'hour', 'min_value'], axis=1, inplace=True)

    return df_with_mins

# TEST THE FUNCTION ABOVE
"""consumption_data =pd.read_csv('../master-data/customers-data/added_type_and_comp.csv')

monthly_df = preprocess_df(consumption_data, Inputs.one_month)
monthly_df['day_of_week'] = monthly_df['Time(Local)'].dt.day_name().astype('category')
monthly_df['hour'] = monthly_df['Time(Local)'].dt.hour

# Compute min and max values
grouped = monthly_df.groupby(['metering_point_id', 'day_of_week', 'hour'])
aggregates = grouped['value'].agg(['min', 'max']).reset_index()

# Create dictionary for fast lookup
lookup_dict = {(row['metering_point_id'], row['day_of_week'], row['hour']): (row['min'], row['max']) for index, row in aggregates.iterrows()}

test_meter = consumption_data["metering_point_id"].iloc[0]

test_hour = monthly_df["Time(Local)"].iloc[110]

test_hour


type(lookup_dict[(test_meter, test_hour.strftime('%A'), 0)][0])"""


def create_meter_objects(consumption_data : pd.DataFrame ,tf : Inputs.GlobalVariables ):
    power_meters = {}
    updated_df = preprocess_df(consumption_data, tf)
    monthly_df = preprocess_df(consumption_data, Inputs.one_month)
    monthly_df['day_of_week'] = monthly_df['Time(Local)'].dt.day_name().astype('category')
    monthly_df['hour'] = monthly_df['Time(Local)'].dt.hour

    # Compute min and max values
    grouped = monthly_df.groupby(['metering_point_id', 'day_of_week', 'hour'])
    aggregates = grouped['value'].agg(['min', 'max']).reset_index()

    # Create dictionary for fast lookup
    lookup_dict = {(row['metering_point_id'], row['day_of_week'], row['hour']): (row['min'], row['max']) for index, row in aggregates.iterrows()}


    for counter, meter_id in enumerate(updated_df["metering_point_id"].unique()):
        if meter_id in monthly_df["metering_point_id"].unique():
            meter_values = updated_df.loc[(updated_df["metering_point_id"] == meter_id)]
            meter_values = meter_values.drop(columns = "metering_point_id")
            #flex_volume = meter_values.copy()
            #flex_volume["value"] = flex_volume["value"]*0.5
            
            directions = ["up", "down", "both"]
            direction_index = rand.randint(0,2) if counter % 5 == 0 else 2
            area = updated_df['area'].loc[(updated_df["metering_point_id"] == meter_id)].iloc[0]
            # response time = mean of the flex_volume divided by the max consumption in the dataset for that meter times a value
            if direction_index == 0: # up
                up_flex_volume = meter_values.copy()
                
                for hour in up_flex_volume["Time(Local)"]:
                    up_flex_volume["value"].loc[up_flex_volume["Time(Local)"] == hour] =  up_flex_volume["value"].loc[up_flex_volume["Time(Local)"] == hour] - lookup_dict[(meter_id, hour.strftime('%A'), hour.hour)][0]

                
                meter = PowerMeter(meter_id = meter_id, response_time = rand.random()*300, up_flex_volume= up_flex_volume , down_flex_volume = [], direction = directions[direction_index], sleep_time = rand.random()*30, consumption_data = meter_values, area = area )
            
            elif direction_index == 1:
                down_flex_volume = meter_values.copy()
                for hour in down_flex_volume["Time(Local)"]:
                    down_flex_volume["value"].loc[down_flex_volume["Time(Local)"] == hour] =  lookup_dict[(meter_id, hour.strftime('%A'), hour.hour)][1] - down_flex_volume["value"].loc[down_flex_volume["Time(Local)"] == hour]
                meter = PowerMeter(meter_id = meter_id, response_time = rand.random()*300, up_flex_volume= [] , down_flex_volume = down_flex_volume, direction = directions[direction_index], sleep_time = rand.random()*30, consumption_data = meter_values, area = area )
            else:
                up_flex_volume = meter_values.copy()
                down_flex_volume = meter_values.copy()
                for hour in up_flex_volume["Time(Local)"]:
                    up_flex_volume["value"].loc[up_flex_volume["Time(Local)"] == hour] =  up_flex_volume["value"].loc[up_flex_volume["Time(Local)"] == hour] - lookup_dict[(meter_id, hour.strftime('%A'), hour.hour)][0]
                    down_flex_volume["value"].loc[down_flex_volume["Time(Local)"] == hour] =  lookup_dict[(meter_id, hour.strftime('%A'), hour.hour)][1] - down_flex_volume["value"].loc[down_flex_volume["Time(Local)"] == hour]
            
                meter = PowerMeter(meter_id = meter_id, response_time = rand.random()*300, up_flex_volume= up_flex_volume , down_flex_volume = down_flex_volume, direction = directions[direction_index], sleep_time = rand.random()*30, consumption_data = meter_values, area = area )

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
    
    """
