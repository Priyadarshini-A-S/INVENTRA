import numpy as np

def calculate_safety_stock(std_demand, lead_days, z_score=1.65):
    # z_score=1.65 for 95% service level
    return np.ceil(z_score * std_demand * np.sqrt(np.maximum(lead_days, 1)))

def calculate_recommended_stock(forecast, safety_stock):
    return np.ceil(forecast + safety_stock)

def calculate_shortage(recommended_stock, current_stock, incoming_stock):
    return np.maximum(0, recommended_stock - (current_stock + incoming_stock))

def calculate_surplus(recommended_stock, current_stock, incoming_stock):
    return np.maximum(0, (current_stock + incoming_stock) - recommended_stock)
