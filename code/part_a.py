import numpy as np
import pandas as pd
from geopy.distance import geodesic

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # radius of Earth in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    res = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)

def nearest_neighbor_optimized(depot_lat, depot_lng, orders):
    # Initialize variables
    remaining_orders = orders.copy()
    current_location = (depot_lat, depot_lng)
    route = [current_location]
    total_distance = 0
    order_ids = []

    # Greedily select the nearest neighbor until all orders are visited
    while len(remaining_orders) > 0:
        min_distance = float('inf')
        nearest_order_idx = None

        # Find the nearest neighbor
        for idx, row in remaining_orders.iterrows():
            distance = haversine(current_location[0], current_location[1], row['lat'], row['lng'])
            if distance < min_distance:
                min_distance = distance
                nearest_order_idx = idx

        # Update route and distance
        nearest_order = remaining_orders.loc[nearest_order_idx]
        route.append((nearest_order['lat'], nearest_order['lng']))
        total_distance += min_distance
        order_ids.append(nearest_order['order_id'])  # Append order ID

        # Update current location and remove visited order
        current_location = (nearest_order['lat'], nearest_order['lng'])
        remaining_orders = remaining_orders.drop(index=nearest_order_idx)

    # Return to depot
    route.append((depot_lat, depot_lng))
    total_distance += haversine(current_location[0], current_location[1], depot_lat, depot_lng)

    return route, total_distance, order_ids

# Load input dataset
input_data = pd.read_csv(r"C:\Users\dell 1\Desktop\Submission\input_datasets\part_a\part_a_input_dataset_5.csv")



# Extract depot coordinates
depot_lat = input_data['depot_lat'][0]
depot_lng = input_data['depot_lng'][0]

# Remove depot coordinates from orders
orders = input_data.drop(columns=['depot_lat', 'depot_lng'])

# Find nearest neighbor route (optimized)
route, total_distance, order_ids = nearest_neighbor_optimized(depot_lat, depot_lng, orders)

# Output route to file
output_data = pd.DataFrame(route[1:-1], columns=['lng', 'lat'])
output_data['depot_lat'] = depot_lat
output_data['depot_lng'] = depot_lng
output_data.insert(0, 'order_id', order_ids)  # Insert order IDs at the beginning
output_data['dlvr_seq_num'] = range(1, len(output_data) + 1)
output_data.to_csv(r"C:\Users\dell 1\Desktop\Submission\output_datasets\part_a\part_a_output_dataset_5.csv", index=False)

