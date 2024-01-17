import requests
import pandas as pd

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
client_secret = "e7bc9a16-c81b-474d-89d7-0801d63acb99"
# Most users will only need the client ID, but if you require access to data that is not open then you need to use the client secret for OAuth2.

# defaults 
timeoffsets= "default"
levels= "default"

qualities=0,1,2,3,4

# Define endpoint and parameters
#The first argument is the endpoint we are going to get data from, in this case the observations endpoint. 
#The next argument is a dictionary containing the parameters that we need to define in order to retrieve data from this endpoint: sources, elements and referencetime.
endpoint = 'https://frost.met.no/observations/v0.jsonld'
parameters = {
    'sources': 'SN18700,SN90450',
    'elements': 'mean(air_temperature P1D),sum(precipitation_amount P1D),mean(wind_speed P1D), cloud_area_fraction, precipitation_amount',
    'referencetime': '2023-06-01/2023-07-01',
}
# Issue an HTTP GET request

def get_weather_data(endpoint : str, parameters : dict):
    r = requests.get(endpoint, parameters, auth=(client_ID,''))
    # Extract JSON data
    json = r.json()# Define endpoint and parameters
    r.status_code
    if r.status_code == 200:
        data = json['data']
        print('Data retrieved from frost.met.no!')
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])


    type(data)
    type(data[0])
    print(data[0])


    df = pd.DataFrame()
    rows = []
    for i in range(len(data)):
        row = pd.DataFrame(data[i]['observations'])
        row['referenceTime'] = data[i]['referenceTime']
        row['sourceId'] = data[i]['sourceId']
        rows.append(row)

    df = pd.concat([df] + rows, ignore_index=True)
    df = df.reset_index()
    df.head()
    df.columns
    df['elementId'].unique()


    # make a shorter and more readable table
    # These additional columns will be kept
    columns = ['sourceId','referenceTime','elementId','value','unit','timeOffset']
    df2 = df[columns].copy()
    # Convert the time value to something Python understands
    df2['referenceTime'] = pd.to_datetime(df2['referenceTime'])
    return df, df2

df, df2 = get_weather_data(endpoint, parameters)

df2.head()

df2['elementId'].unique()
df2.loc[df2['elementId'] == 'precipitation_amount ']
df2.loc[df2['elementId'] == 'mean(air_temperature P1D)']
df2.loc[df2['elementId'] == 'sum(precipitation_amount P1D)']
df2.loc[df2['elementId'] == 'mean(wind_speed P1D)']
cloud_area_rt = df2.loc[df2['elementId'] == 'cloud_area_fraction']
cloud_area_rt.sort_values(by = ['referenceTime'], inplace = True)
cloud_area_rt
cloud_area_rt['referenceTime'].unique()
cloud_area_rt['value'].unique()
cloud_area_rt.loc[cloud_area_rt['value'] == 9.0]
cloud_area_rt.loc[cloud_area_rt['referenceTime'] == pd.Timestamp('2023-06-24 15:00:00+0000', tz='UTC')]
"""want to include cloud_area_fraction as a id. 
oldElementCodes : NN, 
category : Skyer, 
name : Skydekke, 
description :Samla skydekke registreres med kodetall 0-8 som sier hvor mange åttendeler av himmelen som er skydekt (0=skyfritt, 8=helt overskya himmel. 
            Kode -3 eller 9 = mengden av skyer kan ikke bedømmes pga. tåke, snøfokk eller liknende. -3 presenteres som ".") 
unit : octas
status : CF-name
"""
