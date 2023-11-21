import numpy as np
import plotly.graph_objects as go

def plot_flex_volume(H, L, Fu_h_l, Fd_h_l, C_hl):
    """ Plots the total flex volume in both directions for all the meters in the given timeframe

    Args:
        H (list(Timestamp.Timestamp)): The hours in the timeframe
        L (list(power_meter)): list of loads
        Fu_h_l (matrix(float)): The flex volume up for each hour and load
        Fd_h_l (matrix(float)): The flex volume down for each hour and load
        C_hl (matrix_float): The consumption for each hour and load

    Returns:
        None: Plots the flex volume
    """
    fig = go.Figure()
    hourly_flex_vol_up = [np.sum(Fu_h_l[h, l] for l in range(len(L))) for h in range(len(H))]
    hourly_flex_vol_down = [np.sum(Fd_h_l[h, l] for l in range(len(L))) for h in range(len(H))]    
    hourly_tot_cons = [np.sum(C_hl[h, l] for l in range(len(L))) for h in range(len(H))]
    
    fig.add_trace(go.Scatter(x= H, y= hourly_flex_vol_down, mode='lines', name = "Availabale flex volume down"))

    fig.add_trace(go.Scatter(x= H, y= hourly_flex_vol_up,mode='lines', name = "Available flex volume up"))
    fig.add_trace(go.Scatter(x= H, y= hourly_tot_cons, mode='lines', name = "Total consumption"))
    fig.update_layout(title='Plot of flex volume for the meters within the given timeframe', xaxis_title='Time(Local)', yaxis_title='Flex volume [MWh]')
    
    fig.show()
    return None

def plot_prices(price_list, market_names):
    """ Function to plot the hourly prices for each market within the given timeframe

    Args:
        price_list (list(pd.DataFrame)): list of price data for each market
        market_names (list(str)): list of names for each market 

    Returns:
        None: Plots the prices
    """
    fig = go.Figure()
    for df, name in zip(price_list, market_names):
        fig.add_trace(go.Scatter(x= df["Time(Local)"], y=df[df.columns[1]],mode='lines',  name = name))
    fig.update_layout(title='Plot of prices for each market within the given timeframe', xaxis_title='Time(Local)', yaxis_title='Price [EUR/MWh]')
    
    fig.show()
    return None

def plot_volumes(volume_list, market_names):
    """ Function to plot the volumes for each market within the given timeframe

    Args:
        volume_list (list(pd.DataFrame)): list of volume data for each market
        market_names (list(str)): list of names for each market 

    Returns:
        None: Plots the volumes
    """
    fig = go.Figure()
    for df, name in zip(volume_list, market_names):
        fig.add_trace(go.Scatter(x= df["Time(Local)"], y=df[df.columns[1]], mode='lines', name = name))
    fig.update_layout(title='Plot of Volumes for each market within the given timeframe', xaxis_title='Time(Local)', yaxis_title='Volume [MWh]')
    
    fig.show()
    return None


def plot_frequency(frequency_df):
    """ Plots the frequency in the nordic grid for the given timeframe

    Args:
        frequency_df (pd.DataFrame): dataframe of the frequency in the nordic grid

    Returns:
        None: plots the frequency
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x= frequency_df[frequency_df.columns[0]], y=frequency_df[frequency_df.columns[1]], mode='lines', name = "Frequency"))
    fig.update_layout(title='Plot of the frequency in the nordic grid', xaxis_title='Time(Local)', yaxis_title='Frequency [Hz]')
    fig.show()
    return None

def plot_capacity_market_activation(Va_hm, market_names, H):
    """ Plots the activation volumes for each market within the given timeframe

    Args:
        Va_hm (dict): dictionary of the activation volumes for each market
        market_names (list(str)): list of names for each market
        H (list(Timeframe.Timeframe)): list of hours in the timeframe

    Returns:
        None: plots the activation volumes
    """
    fig = go.Figure()

    for m, market_name in enumerate(market_names):
        fig.add_trace(go.Scatter(x= H, y= [Va_hm[h,m] for h in range(len(H))], mode='lines', name = market_name))
        
    fig.update_layout(title='Plot activation volumes' , xaxis_title='Time(Local)', yaxis_title='Activation volume [MWh]')
    fig.show()
    return None


        
    





