import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random as rand
from code_map import Utils


class PowerMeter:
    def __init__(self, meter_id, response_time, flex_volume, direction, sleep_time, consumption_data, area):
        self.response_time = response_time
        self.flex_volume = flex_volume
        self.direction = direction
        self.sleep_time = sleep_time
        self.consumption_data = consumption_data
        self.meter_id = meter_id
        self.area = area
       
        
        
consumption_data =pd.read_csv('customers-data/added_type_and_comp.csv')

#consumption_data.head()

#len(consumption_data)

production_meters_df = consumption_data.loc[consumption_data['meter_type'] == 'Production']

combined_meters_df = consumption_data.loc[consumption_data['meter_type'] == 'Combined']

combined_meters_df['metering_point_id'].unique()
"""version_variables = Inputs.one_month

start_date = pd.Timestamp(version_variables.year, version_variables.start_month, version_variables.start_day, version_variables.start_hour).tz_localize('Europe/Oslo')
start_date
end_date = pd.Timestamp(version_variables.year, version_variables.end_month, version_variables.end_day, version_variables.end_hour).tz_localize('Europe/Oslo')
end_date
consumption_data["start_time_local"] = pd.to_datetime(consumption_data["start_time_local"])
consumption_data["start_time_local"] = consumption_data["start_time_local"].dt.tz_convert('Europe/Oslo')
consumption_data["start_time_local"]
hours_in_timehorizone = (end_date - start_date).days*24 + (end_date - start_date).seconds/3600

consumption_data = consumption_data.loc[(consumption_data["start_time_local"] >= start_date) & (consumption_data["start_time_local"] <= end_date)]

count_hours_df = consumption_data.groupby("metering_point_id")["value"].agg(["count"])
count_hours_df.to_csv("count_hours.csv")

missing_hours_df = count_hours_df.loc[(count_hours_df["count"] < hours_in_timehorizone)]
missing_hours_df.to_csv("missing_hours.csv")


each_hour_count = consumption_data.groupby("start_time_local").agg("count")
each_hour_count.to_csv("count.csv")"""



def preprocess_df(df, version_variables : Utils.GlobalVariables):
    start_date = pd.Timestamp(version_variables.year, version_variables.start_month, version_variables.start_day, version_variables.start_hour).tz_localize('Europe/Oslo')
    end_date = pd.Timestamp(version_variables.year, version_variables.end_month, version_variables.end_day, version_variables.end_hour).tz_localize('Europe/Oslo')
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

updated_df = preprocess_df(consumption_data, Utils.one_day)

#len(updated_df["metering_point_id"].unique())

updated_df.head()

#updated_df["metering_point_id"].iloc[0]

def create_meter_objects(updated_df):
    power_meters = {}

    for counter, meter_id in enumerate(updated_df["metering_point_id"].unique()):
        meter_values = updated_df.loc[(updated_df["metering_point_id"] == meter_id)]
        meter_values = meter_values.drop(columns = "metering_point_id")
        flex_volume = meter_values.copy()
        flex_volume["value"] = flex_volume["value"]*0.5
        
        directions = ["up", "down", "both"]
        direction_index = rand.randint(0,2) if counter % 5 == 0 else 2
        area = updated_df['area'].loc[(updated_df["metering_point_id"] == meter_id)].iloc[0]
        # response time = mean of the flex_volume divided by the max consumption in the dataset for that meter times a value
        
        meter = PowerMeter(meter_id = meter_id, response_time = rand.random()*300 , flex_volume = flex_volume, direction = directions[direction_index], sleep_time = rand.random()*30, consumption_data = meter_values, area = area )
        
        power_meters[meter_id] = meter
    return power_meters

power_meters = create_meter_objects(updated_df)
    
"""test_meter = list(power_meters.keys())[9]
test_meter
power_meters[test_meter].direction
"""
        
