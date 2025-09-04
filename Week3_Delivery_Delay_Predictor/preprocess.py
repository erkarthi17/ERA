def encode_features(data):
    traffic_map = {"Low": 0, "Medium": 1, "High": 2}
    weather_map = {"Clear": 0, "Rain": 1, "Snow": 2, "Fog": 3}
    vehicle_map = {"Van": 0, "Truck": 1, "Bike": 2}
    time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}

    return [
        data["distance_km"],
        traffic_map[data["traffic_level"]],
        weather_map[data["weather"]],
        vehicle_map[data["vehicle_type"]],
        time_map[data["time_of_day"]]
    ]
