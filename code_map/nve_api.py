import requests
import pandas as pd



#curl -X GET "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData" -H "accept: application/json"

#def get_fyllingsgrad_df(year : int, area : int = 10):
"""Function to get a dataframe for the wanted year and area from the NVE API

Args:
    year (int): the wanted year
    area (int, optional): if the dataframe for only a specific area is wanted this are should be added as ant int here. Defaults to 10 which corresponds to all areas

Returns:
    fyllingsgrad_df (pd.DataFrame): dataframe with the wanted data
"""
<<<<<<< HEAD
def get_fyllinsgrad_df(areas = ["NO1", "NO2", "NO3", "NO4", "NO5"], year : int = 2023) -> pd.DataFrame:
    """ Function to get data for fyllingsgrad from the NVE API. 
        The data is returned as a pandas dataframe with the following columns: 
        Uke, Dato, Fyllingsgrad, Område, Endring fyllinsgrad, Kapasitet TWh, Fylling TWh

        The Dato column is converted to datetime format and the timezone is set to Europe/Oslo
        Fyllingsgrad and Endring fyllingsgrad are in percent which means that it doesnt need to be normalized.
        Kapasitet TWh and Fylling TWh are in TWh which meand they needs to be normalized if they should be used.
    Args:
        areas (list, optional): List of the areas wanted. Defaults to ["NO1", "NO2", "NO3", "NO4", "NO5"].
        year (int, optional): For which year the data is wanted. Defaults to 2023.

    Returns:
        pd.DataFrame: Dataframe with the wanted data
    """

    request_url = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData"

    response = requests.get(request_url)

    nve_dict_list = response.json()
    new_list = []
    for dict in nve_dict_list:
        if dict["iso_aar"] == year and dict["omrType"] == "EL":
            new_list.append(dict)
        else:
            continue

    fyllingsgrad_df = pd.DataFrame(columns = ["Uke", "Dato", "Fyllingsgrad", "Område", "Endring fyllinsgrad", "Kapasitet TWh", "Fylling TWh"])

    for dict in new_list:
        fyllingsgrad_df.loc[len(fyllingsgrad_df)] = [dict["iso_uke"], dict["dato_Id"], dict["fyllingsgrad"], dict["omrnr"], dict["endring_fyllingsgrad"], dict["kapasitet_TWh"], dict["fylling_TWh"], ]
        #print(len(fyllingsgrad_df))

    fyllingsgrad_df.sort_values(by = ["Uke", "Område"], inplace = True)
    fyllingsgrad_df["Dato"]
    date_format = '%Y-%m-%d'
    fyllingsgrad_df["Dato"] = pd.to_datetime(fyllingsgrad_df["Dato"], format = date_format, utc = True)
    fyllingsgrad_df["Dato"] = fyllingsgrad_df["Dato"].dt.tz_convert('Europe/Oslo')
    int_to_area_dict = {1 : "NO1", 2 : "NO2", 3 : "NO3", 4 : "NO4", 5 : "NO5"}
    fyllingsgrad_df["Område"] = fyllingsgrad_df["Område"].apply(lambda x : int_to_area_dict[x])

    if len(areas) >= 5:
        return fyllingsgrad_df
    else:
        fyllingsgrad_df = fyllingsgrad_df.loc[fyllingsgrad_df["Område"].isin(areas)]
        return fyllingsgrad_df.reset_index(drop = True)

#get_fyllinsgrad_df(areas = ["NO5"])
=======
year= 2023

request_url = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData"

response = requests.get(request_url)

nve_dict_list = response.json()
new_list = []
for dict in nve_dict_list:
    if dict["iso_aar"] == year and dict["omrType"] == "EL":
        new_list.append(dict)
    else:
        continue

#new_list[0].keys()
#len(new_list)

fyllingsgrad_df = pd.DataFrame(columns = ["Uke", "Dato", "Fyllingsgrad", "Område", "Endring fyllinsgrad", "Kapasitet TWh", "Fylling TWh"])

for dict in new_list:
    fyllingsgrad_df.loc[len(fyllingsgrad_df)] = [dict["iso_uke"], dict["dato_Id"], dict["fyllingsgrad"], dict["omrnr"], dict["endring_fyllingsgrad"], dict["kapasitet_TWh"], dict["fylling_TWh"], ]
    print(len(fyllingsgrad_df))


fyllingsgrad_df.sort_values(by = ["Uke"], inplace = True)
fyllingsgrad_df["Dato"]
date_format = '%Y-%m-%d'
fyllingsgrad_df["Dato"] = pd.to_datetime(fyllingsgrad_df["Dato"], format = date_format, utc = True)
fyllingsgrad_df["Dato"] = fyllingsgrad_df["Dato"].dt.tz_convert('Europe/Oslo')

    
fyllingsgrad_df
>>>>>>> 87f793b74b2c8f4bca7853f5a8c4e88d08c0a54b


#fdf = get_fyllingsgrad_df(year =2023)

# This is the code for the area request - dont think it will be usefull
"""area_request = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOmråder"
area_response = requests.get(area_request)
area_dict_list = area_response.json()
len(area_dict_list) # 1
area_dict = area_dict_list[0]
area_dict.keys()
area_dict["land"]
area_dict["elspot"]
area_dict["vassdrag"]"""