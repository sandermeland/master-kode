import requests
import pandas as pd
import numpy as np

from code_map import timeframes, rl_utils, utils


"""
All requests must (if possible) include an identifying User Agent-string (UA) in the request with the application/domain name, optionally version number. 
You should also include a company email address or a link to the company website where we can find contact information. 
If we cannot contact you in case of problems, you risk being blocked without warning.

Examples of valid user-Agents : 
"acmeweathersite.com support@acmeweathersite.com"
"AcmeWeatherApp/0.9 github.com/acmeweatherapp"
"""

# For the FROST API
client_ID = "ccb8bad7-7d8f-4830-ba07-8228703e60e6"
# Most users will only need the client ID, but if you require access to data that is not open then you need to use the client secret for OAuth2.

# defaults 
"""timeoffsets= "default"
levels= "default"

qualities=0,1,2,3,4"""

# Define endpoint and parameters
#The first argument is the endpoint we are going to get data from, in this case the observations endpoint. 
#The next argument is a dictionary containing the parameters that we need to define in order to retrieve data from this endpoint: sources, elements and referencetime.

# Issue an HTTP GET request


def get_sources_with_areas_from_table():
    """ Function to get the sources from the stations from the table I downloaded earlier
        These are not included in the API approach

    Returns:
        [type]: [description]
    """
    stations_from_table = ["SN50540", "SN4780", "SN39150", "SN90450", "SN68230"]
    # Bergen (Florida), Gardermoen, Kristiansand - Sømskleiva, Tromsø, Trondheim - Risvollan
    table_sa_dict = {"SN50540" : "NO5", "SN4780" : "NO1", "SN39150" : "NO2", "SN90450" : "NO4", "SN68230" : "NO3"}
    return  stations_from_table, table_sa_dict


def get_sources_with_areas(areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]):
    """ Function to get the sources from the frost API and add the area to the dataframe as well as the rest of the information
        The area is added based on the municipality of the station
        The api provides the following information for each station:
        "name", "country", "validFrom", "county", "municipality", "coordinates"


    Returns:
        df (pd.DataFrame): dataframe with the wanted data where the area is added
        station_area_dict (dict): dictionary with station_id as key and area as value
    """
    sources_endpoint = "https://frost.met.no/sources/v0.jsonld"

    sources = requests.get(sources_endpoint, auth=(client_ID,''))
    json = sources.json()

    sources.status_code
    data = json['data']

    #print(data[0])
    #len(data)

    df = pd.DataFrame(columns = ["station_id", "name", "country", "validFrom" ,"county", "municipality", "coordinates"])
    for dict in data:
        key_list = ["id", "name", "country", "validFrom", "county", "municipality", 'geometry']
        if all(key in dict.keys() for key in key_list):
            if "validTo" not in dict.keys():
                df.loc[len(df)] = [dict["id"], dict["name"], dict["country"], dict["validFrom"], dict["county"], dict["municipality"], dict["geometry"]["coordinates"]]
            #print(len(df))
        else:
            continue


    municipality_area_dict = {"OSLO" : "NO1", "ASKIM" : "NO1", "KONGSBERG" : "NO1", "SARPSBORG" : "NO1", "HØNEFOSS" : "NO1", "RENDALEN" : "NO1",
                            "STAVANGER" :"NO2", "KRISTIANSAND" : "NO2", "HAUGESUND" : "NO2", "EGERSUND" : "NO2", "PORSGRUNN" : "NO2",
                            "BERGEN" : "NO5", "VOSS" : "NO5", "ASKØY" : "NO5", "FLORØ" : "NO5", "SOGNDAL" : "NO5", "GEILO" : "NO5", "ÅL" : "NO5",
                            "TRONDHEIM" : "NO3", "STEINKJER" : "NO3", "NAMSOS" : "NO3", "RØROS" : "NO3", "MOLDE" : "NO3", "KRISTIANSUND" : "NO3",
                            "TROMSØ" : "NO4", "BODØ" : "NO4", "NARVIK" : "NO4", "ALTA" : "NO4", "HAMMERFEST" : "NO4", "VADSØ" : "NO4"}

    if len(areas) < 5:
        # remove the key-value pairs where the value is not in the areas list
        municipality_area_dict = {key:val for key, val in municipality_area_dict.items() if val in areas}
    

    df["area"] = df["municipality"].map(municipality_area_dict)
                            
    df = df.loc[df["area"].notna()]
    station_area_dict = df.set_index("station_id").to_dict()["area"]

    return df, station_area_dict



def select_spread_stations(area_df : pd.DataFrame, n : int):
    """ Function to select stations based on spread of coordinates

    Args:
        area_df (pd.DataFrame): dataframe of stations filtered for the area
        n (int): amount of station wanted from the given area

    Returns:
        area_df (pd.DataFrame): dataframe of n stations filtered for the area
    """
    # Calculate the variance of coordinates
    df = area_df.copy()
    df['coord_variance'] = df['coordinates'].apply(lambda x: np.var(x))
    # Sort by variance and pick top n
    return df.sort_values(by='coord_variance', ascending=False).head(n)['station_id'].tolist()


def get_station_ids(n=5, areas = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']):
    """ Function to get the n station ids for the wanted areas with the highest spread of coordinates

    Args:
        n (int, optional): Number of stations within each area. Defaults to 5.
        areas (list, optional): The areas for which station to be added. Defaults to ['NO1', 'NO2', 'NO3', 'NO4', 'NO5'].

    Returns:
        station_list (list): list of station ids. n for each area in areas with the highest spread of coordinates
        sa_dict (dict) : dictionary with station_id as key and area as value
    """
    station_df, sa_dict = get_sources_with_areas(areas = areas)
    selected_stations = []

    # Loop through the areas 
    for area in areas:
        area_stations = station_df[station_df['area'] == area]
        selected_stations.extend(select_spread_stations(area_stations, n = n-1))

    #remove key-value pairs from the sa_dict where the key is not in the selected_stations list
    sa_dict = {key:val for key, val in sa_dict.items() if key in selected_stations}
    table_stations, table_sa_dict = get_sources_with_areas_from_table()
    for area in areas:
        for station in table_stations:
            if table_sa_dict[station] == area:
                selected_stations.append(station)
                sa_dict[station] = area
                break
    return selected_stations, sa_dict



def get_weather_data(tf : timeframes.TimeFrame, sa_dict: dict, wanted_sources : list = ['SN18700', 'SN90450'], wanted_elements : list = ["mean(air_temperature P1D)", "sum(precipitation_amount P1D)", "mean(wind_speed P1D)", "cloud_area_fraction"]):
    """
    Args:
        start_month (int): start month of the wanted data
        end_month (int): end month of the wanted data
        start_day (int): start day of the wanted data
        end_day (int): end day of the wanted data
        start_year (int): start year of the wanted data
        end_year (int): end year of the wanted data
        sa_dict (dict): dictionary with station_id as key and area as value
        wanted_sources (list, optional): list of wanted station_ids. Defaults to ['SN18700', 'SN90450'].
        wanted_elements (list, optional): list of wanted elements. Defaults to ["mean(air_temperature P1D)", "sum(precipitation_amount P1D)", "mean(wind_speed P1D)", "cloud_area_fraction"].

    Returns:
        df1 (pd.DataFrame): dataframe with extended information
        df2 (pd.DataFrame): dataframe with the wanted data. Filtered version of df1
    
    """
    endpoint = 'https://frost.met.no/observations/v0.jsonld'

    parameters = {
    'sources': ",".join(wanted_sources),
    'elements': ",".join(wanted_elements),
    'referencetime': f'{tf.year}-{tf.start_month}-{tf.start_day}/{tf.year}-{tf.end_month}-{tf.end_day}',
    }

    r = requests.get(endpoint, parameters, auth=(client_ID,''))

    # Extract JSON data
    json = r.json()# Define endpoint and parameters
    r.status_code
    if r.status_code == 200:
        data = json['data']
        #print('Data retrieved from frost.met.no!')
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])
        return None, None


    df = pd.DataFrame()
    rows = []
    for i in range(len(data)):
        row = pd.DataFrame(data[i]['observations'])
        row['referenceTime'] = data[i]['referenceTime']
        row['sourceId'] = data[i]['sourceId']
        rows.append(row)

    df = pd.concat([df] + rows, ignore_index=True)
    # add area to the dataframe by using the sa_dict where the key is the station_id and the value is the area

    df = df.reset_index()
    df["referenceTime"] = pd.to_datetime(df["referenceTime"], utc = True)
    df["referenceTime"] = df["referenceTime"].dt.tz_convert('Europe/Oslo')
  
    # make a shorter and more readable table
    # These additional columns will be kept
    columns = ['sourceId','referenceTime','elementId','value','unit']#,'timeOffset']
    df2 = df[columns].copy()
    df2["sourceId"] = df2["sourceId"].str[:-2]
    df2['area'] = df2['sourceId'].map(sa_dict)

    # Convert the time value to something Python understands
    df2['referenceTime'] = pd.to_datetime(df2['referenceTime'])
    return df, df2

"""station, sa_dict = get_sources_with_areas()

table_stations, table_sa_dict = get_sources_with_areas_from_table()

_,df2 = get_weather_data(tf = timeframes.one_month, sa_dict= table_sa_dict, wanted_sources= table_stations, wanted_elements= ["mean(air_temperature P1D)", "sum(precipitation_amount P1D)", "mean(wind_speed P1D)", "cloud_area_fraction"] )



len(df2["sourceId"].unique())

df2['elementId'].unique() # there is no stations that has precipitation amount !! #TODO : find out why none of the stations has precipitation amount. sum(precipitation_amount P1D) is working, so might be a problem with the elementId
df2
df2["referenceTime"].unique()

len(df2.loc[df2['elementId'] == 'sum(precipitation_amount P1D)']) # = 10 which is correct for 5 stations and 2 days
df2.loc[df2['elementId'] == 'sum(precipitation_amount P1D)']
df2.loc[df2['elementId'] == 'mean(air_temperature P1D)']
df2.loc[df2['elementId'] == 'sum(precipitation_amount P1D)']
df2.loc[df2['elementId'] == 'mean(wind_speed P1D)']
cloud_area_rt = df2.loc[df2['elementId'] == 'cloud_area_fraction']
cloud_area_rt.sort_values(by = ['referenceTime'], inplace = True)
cloud_area_rt
cloud_area_rt['area'].unique()
cloud_area_rt.loc[cloud_area_rt['area'] == "NO5"]
cloud_area_rt['referenceTime'].unique()
cloud_area_rt['value'].unique()
cloud_area_rt.loc[cloud_area_rt['value'] == 9.0]
cloud_area_rt.loc[cloud_area_rt['referenceTime'] == pd.Timestamp('2023-06-24 15:00:00+0000', tz='UTC')]"""

def get_wanted_weather_values(orig_df, elementId, area):
    # groupby elementid and area and then find hourly mean for each day
    df = orig_df.copy()
    df = df.loc[df['elementId'] == elementId]
    df = df.loc[df['area'] == area]
    df.set_index('referenceTime', inplace=True)
    if elementId == 'sum(precipitation_amount P1D)' or elementId == 'cloud_area_fraction':
        mean_values_per_hour = df.resample('D')["value"].mean()
    else:
        mean_values_per_hour = df.resample('H')["value"].mean()
    #mean_values_per_hour = mean_values_per_hour.reset_index()
    return mean_values_per_hour

#get_wanted_weather_values(df2, elementId= 'cloud_area_fraction', area = 'NO5')


def get_normalized_weather_dfs(reference_tf : timeframes.TimeFrame, usage_tf : timeframes.TimeFrame, norm_method = "min_max", areas = ["NO1", "NO2", "NO3", "NO4", "NO5"],  elements  = ['air_temperature', 'sum(precipitation_amount P1D)', 'wind_speed', 'cloud_area_fraction']):
    stations, sa_dict = get_station_ids(n=6, areas = areas)
    _, reference_df = get_weather_data(tf = reference_tf, wanted_sources= stations, sa_dict = sa_dict ,wanted_elements= elements )
<<<<<<< HEAD
=======
    usage_hours = utils.get_timestamps(usage_tf)
>>>>>>> 87f793b74b2c8f4bca7853f5a8c4e88d08c0a54b
    noramlized_dfs_dict = {}
    for element in elements:
        for area in areas:
            #print(f"element : {element}, area : {area}")
            element_df = get_wanted_weather_values(reference_df, elementId= element, area = area)
            
            if norm_method == "min_max":
<<<<<<< HEAD
                #print(element_df)
=======
                print(element_df)
>>>>>>> 87f793b74b2c8f4bca7853f5a8c4e88d08c0a54b
                min_val = element_df.min()
                max_val = element_df.max()
                element_df = (element_df - min_val) / (max_val - min_val)
            else: #z-score normali1
                mean_val = element_df.mean()
                std_val = element_df.std()
                element_df = (element_df - mean_val) / std_val
<<<<<<< HEAD
            usage_df = element_df.loc[element_df.index.isin(utils.get_timestamps(usage_tf))]
=======

            usage_df = element_df.loc[element_df.index.isin(usage_hours)]
>>>>>>> 87f793b74b2c8f4bca7853f5a8c4e88d08c0a54b
            noramlized_dfs_dict[(element, area)]  = usage_df

    return noramlized_dfs_dict


<<<<<<< HEAD
"""weather_dict = get_normalized_weather_dfs(reference_tf= timeframes.one_month, usage_tf = timeframes.one_week, areas = ["NO5"])

for i in weather_dict.keys():
=======
#weather_dict = get_normalized_weather_dfs(reference_tf= timeframes.one_month, usage_tf = timeframes.one_week, areas = ["NO5"])

"""for i in weather_dict.keys():
>>>>>>> 87f793b74b2c8f4bca7853f5a8c4e88d08c0a54b
    print(i)
    print(weather_dict[i])"""


"""want to include cloud_area_fraction as a id. 
oldElementCodes : NN, 
category : Skyer, 
name : Skydekke, 
description :Samla skydekke registreres med kodetall 0-8 som sier hvor mange åttendeler av himmelen som er skydekt (0=skyfritt, 8=helt overskya himmel. 
            Kode -3 eller 9 = mengden av skyer kan ikke bedømmes pga. tåke, snøfokk eller liknende. -3 presenteres som ".") 
unit : octas
status : CF-name
"""
