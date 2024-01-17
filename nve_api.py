import requests
from requests import Request, Session


#curl -X GET "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData" -H "accept: application/json"

request_url = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData"

response = requests.get(request_url)

nve_dict_list = response.json()

new_list = []
for dict in nve_dict_list:
    if dict["iso_aar"] > 2022:
        new_list.append(dict)
len(nve_dict_list)
len(new_list)

print(new_list[0])

import pandas as pd
fyllingsgrad_df = pd.DataFrame(columns = ["Uke", "Dato", "Fyllingsgrad", "Område"])

for dict in new_list:
    fyllingsgrad_df.loc[len(fyllingsgrad_df)] = [dict["iso_uke"], dict["dato_Id"], dict["fyllingsgrad"], dict["omrnr"]]
    print(len(fyllingsgrad_df))

fyllingsgrad_df.sort_values(by = ["Uke"], inplace = True)

fyllingsgrad_df
