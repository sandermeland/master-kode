import pandas as pd 
from code_map import timeframes


#weather_data = pd.read_csv('../master-data/weather_data/weather_5_meters.csv', sep=';')

def preprocess_weather_data(df : pd.DataFrame, tf : timeframes.TimeFrame) -> pd.DataFrame:
    #df.drop(df.loc[df["Navn"] ==  'Data er gyldig per 06.12.2023 (CC BY 4.0), Meteorologisk institutt (MET)'], inplace=True)
    df = df.copy()
    df.rename(columns={'Tid(norsk normaltid)' : 'Time (Local)', \
                       "Navn" : "station_name", \
                       "Stasjon" : "station", \
                       "Lufttemperatur" : "air_temp", \
                       "Vindretning" : "wind_direction" , \
                       "Nedbør (1 t)" : "precipitation"}, inplace=True)
    # convert from str to float
    for column_name in ['precipitation',  'air_temp']:
        df[column_name] = df[column_name].str.replace(',', '.')
        df[column_name] = df[column_name].astype(float)

    df["area"] = df["station_name"].apply(lambda x : "NO1" if x == 'Gardermoen' else "NO2" if x == 'Kristiansand - Sømskleiva' else "NO3" if x == 'Trondheim - Risvollan' else "NO4" if x == 'Tromsø' else "NO5" if x == 'Bergen - Florida' else "None")
    # get this format : 01.01.2022 01:00
    format = '%d.%m.%Y %H:%M'
    df['Time (Local)'] = pd.to_datetime(df['Time (Local)'], format= format, utc=True)
    df['Time (Local)'] = df['Time (Local)'].dt.tz_convert('Europe/Oslo')
    df.sort_values(by=['Time (Local)'], inplace=True)
    df = df.loc[(df["Time (Local)"] >= pd.Timestamp(tf.year, tf.start_month, tf.start_day, tf.start_hour, tz='Europe/Oslo')) & (df["Time (Local)"] <= pd.Timestamp(tf.year, tf.end_month, tf.end_day, tf.end_hour, tz='Europe/Oslo')) ]
    df.reset_index(drop=True, inplace= True)

    df.drop(['station_name', 'station'], axis=1, inplace=True)
    return df

#weather_data = preprocess_weather_data(weather_data)

def get_weather_data(tf : timeframes.TimeFrame, areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]):
    weather_data = pd.read_csv('../master-data/weather_data/weather_5_meters.csv', sep=';')
    df = preprocess_weather_data(weather_data, tf)
    if len(areas) < 5:
        return df.loc[df["area"].isin(areas)]
    else:
        return df

