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