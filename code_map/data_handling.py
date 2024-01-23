import pickle
import pandas as pd
import numpy as np



def save_collections(L,M,F,H,freq_data,power_meter_dict,consumption_data,L_u,L_d,Fu_h_l,Fd_h_l,R_h_l,P_h_m,Vp_h_m,Vm_m,R_m,dominant_directions,Ir_hlm,Ia_hlm,Va_hm,compatible_list, pkl_filename):
    """ A function to store values to a pickle file. The pickle file can later be used to load the collections and this should be way faster than creating the collections from scratch.

    Args:
        L (_type_): _description_
        M (_type_): _description_
        F (_type_): _description_
        H (_type_): _description_
        freq_data (_type_): _description_
        power_meter_dict (_type_): _description_
        consumption_data (_type_): _description_
        L_u (_type_): _description_
        L_d (_type_): _description_
        Fu_h_l (_type_): _description_
        Fd_h_l (_type_): _description_
        R_h_l (_type_): _description_
        P_h_m (_type_): _description_
        Vp_h_m (_type_): _description_
        Vm_m (_type_): _description_
        R_m (_type_): _description_
        dominant_directions (_type_): _description_
        Ir_hlm (_type_): _description_
        Ia_hlm (_type_): _description_
        Va_hm (_type_): _description_
        compatible_list (_type_): _description_
        pkl_filename (_type_): _description_

    """
    collection_list = [L, M, F, H, freq_data, power_meter_dict, consumption_data, L_u, L_d, Fu_h_l, Fd_h_l, R_h_l, P_h_m, Vp_h_m, Vm_m, R_m, dominant_directions, Ir_hlm, Ia_hlm, Va_hm, compatible_list]
    name_list = ["L", "M", "F", "H", "freq_data", "power_meter_dict", "consumption_data", "L_u", "L_d", "Fu_h_l", "Fd_h_l", "R_h_l", "P_h_m", "Vp_h_m", "Vm_m", "R_m", "dominant_directions", "Ir_hlm", "Ia_hlm", "Va_hm", "compatible_list"]
    # Assuming your collections are named collection1, collection2, ..., collection12
    all_collections = {name : collection for name, collection in zip(name_list, collection_list)}

    with open(pkl_filename, 'wb') as file:
        pickle.dump(all_collections, file)
    return None

def load_collections(load_path):
    """ A function to load all the collections from a pickle file. This should be way faster than creating the collections from scratch.
    This is the correct order for the collections:
    L, M, F, H, freq_data, power_meter_dict, consumption_data, L_u, L_d, Fu_h_l, Fd_h_l, R_h_l, P_h_m, Vp_h_m, Vm_m, R_m, dominant_directions, Ir_hlm, Ia_hlm, Va_hm, compatible_list

    Args:
        load_path (str): path to the pickle file where the wanted collections are stored

    Returns:
        _type_: _description_
    """
    with open(load_path, 'rb') as file:
        loaded_collections = pickle.load(file)

    # Now you can access each collection individually, e.g.,
    L = loaded_collections["L"]
    M = loaded_collections["M"]
    F = loaded_collections["F"]
    H = loaded_collections["H"]
    freq_data = loaded_collections["freq_data"]
    power_meter_dict = loaded_collections["power_meter_dict"]
    consumption_data = loaded_collections["consumption_data"]
    L_u = loaded_collections["L_u"]
    L_d = loaded_collections["L_d"]
    Fu_h_l = loaded_collections["Fu_h_l"]
    Fd_h_l = loaded_collections["Fd_h_l"]
    R_h_l = loaded_collections["R_h_l"]
    P_h_m = loaded_collections["P_h_m"]
    Vp_h_m = loaded_collections["Vp_h_m"]
    Vm_m = loaded_collections["Vm_m"]
    R_m = loaded_collections["R_m"]
    dominant_directions = loaded_collections["dominant_directions"]
    Ir_hlm = loaded_collections["Ir_hlm"]
    Ia_hlm = loaded_collections["Ia_hlm"]
    Va_hm = loaded_collections["Va_hm"]
    compatible_list = loaded_collections["compatible_list"]
    return L, M, F, H, freq_data, power_meter_dict, consumption_data, L_u, L_d, Fu_h_l, Fd_h_l, R_h_l, P_h_m, Vp_h_m, Vm_m, R_m, dominant_directions, Ir_hlm, Ia_hlm, Va_hm, compatible_list