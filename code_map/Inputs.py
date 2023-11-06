import pandas as pd
import os
from zoneinfo import ZoneInfo
from functools import reduce

import pytz

class GlobalVariables():
    def __init__(self, year,start_month, end_month, start_day, end_day, start_hour, end_hour, area): 
        self.year = year
        self.start_month = start_month
        self.end_month = end_month
        self.start_day = start_day
        self.end_day = end_day
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.area = area


one_hour = GlobalVariables(year = 2023, start_month = 6, end_month = 6, start_day = 26, end_day = 26, start_hour = 15, end_hour = 16, area = "NO5") # it may be possible to start from hour 14


one_day = GlobalVariables(year = 2023, start_month = 6, end_month = 6, start_day = 26, end_day = 26, start_hour = 0, end_hour = 23, area = "NO5")

half_month =  GlobalVariables(year = 2023, start_month = 6, end_month = 6, start_day = 14, end_day = 30, start_hour = 0, end_hour = 23, area = "NO5")

one_month = GlobalVariables(year = 2023, start_month = 6, end_month = 6, start_day = 1, end_day = 30, start_hour = 0, end_hour = 23, area = "NO5")


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
    freq_df["Time"] = pd.to_datetime(freq_df["Time"], format = "%Y-%m-%d %H:%M:%S")
    freq_df["Time"] = freq_df["Time"].dt.tz_localize("Europe/Oslo")
    start_datetime = pd.Timestamp(year = tf.year, month= tf.start_month, day=tf.start_day, hour= tf.start_hour, tz = "Europe/Oslo") #Europe/Oslo    
    end_datetime = pd.Timestamp(year = tf.year, month= tf.end_month, day=tf.end_day, hour= tf.end_hour, tz = "Europe/Oslo")
    filtered_df = freq_df[(freq_df["Time"] >= start_datetime) & (freq_df["Time"] <= end_datetime)]
    filtered_df.sort_values(by = "Time", inplace = True)
    filtered_df.reset_index(inplace = True, drop = True)
    
    return filtered_df


#freq_df = get_frequency_data(tf = one_day, freq_directory = '../master-data/frequency_data/2023-06')
#freq_df.head()

def get_afrr_activation_data(tf : GlobalVariables, afrr_directory : str, direction : str):
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

        
    #afrr_df = pd.concat(afrr_dfs, ignore_index= True)
    #print(afrr_df.head())
    
   
    #afrr_df = afrr_df[afrr_df["Time"] != "Balancing Time Unit (Automatic Frequency Restoration Reserve (aFRR))"]
    #print(afrr_df["Time"].str.slice(0, 16))
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
    
    """ generation_data = filtered_df.loc[filtered_df["Source"] == "Generation"]
    load_data = filtered_df.loc[filtered_df["Source"] == "Load"]
    
    generation = (generation_data.applymap(lambda x: x > 0 if isinstance(x, float) else False)).any().any()
    load = (load_data.applymap(lambda x: x > 0 if isinstance(x, float) else False)).any().any()
    
    if not generation and not load:
        filtered_df = filtered_df.loc[filtered_df["Source"] == "Not specified"]"""
    
    
    filtered_df.sort_values(by = "Time", inplace = True)
    filtered_df.reset_index(inplace = True, drop = True)
    filtered_df.iloc[:, 1:6] = filtered_df.iloc[:, 1:6].astype(float)

    
    return filtered_df

afrr_activation = get_afrr_activation_data(tf = one_day, afrr_directory = '../master-data/aFRR_activation/', direction = "Up")
afrr_activation.columns
afrr_activation[afrr_activation.columns[1:6]]
#afrr_activation.to_csv("only_up_activations.csv")

afrr_activation.iloc[:, 1:6] = afrr_activation.iloc[:, 1:6].astype(float)

afrr_activation.columns



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

