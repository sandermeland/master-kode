import pandas as pd
import os

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

