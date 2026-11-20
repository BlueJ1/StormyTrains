"""Column names of the GFS forecast inputs (build_forecast_weather.py, decision D17).

Kept apart from the builder so that the feature lists, the loader and the 1.5
representations can import them without importing the builder.
"""
FORECAST = ("forecast_origin_temperature_c", "forecast_destination_temperature_c",
            "forecast_route_temp_max_c", "forecast_route_temp_min_c",
            "forecast_route_precip_max_mm", "forecast_route_precip_total_mm",
            "forecast_route_wind_max_ms", "forecast_route_gust_max_ms",
            "forecast_route_snow_depth_cm")
FORECAST_LAG = tuple(f"{column}_lag24h" for column in FORECAST)
FORECAST_CHANGE = tuple(f"{column}_change24h" for column in FORECAST)
