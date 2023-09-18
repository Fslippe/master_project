from datetime import datetime, timedelta 

def convert_to_day_of_year(date_str):
    # Parse the date
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    # Convert to datetime object
    date_obj = datetime(year, month, day)

    # Get the day of the year
    day_of_year = date_obj.timetuple().tm_yday

    # Return in the desired format
    return f"{year}{day_of_year:03d}"  # Using :03d to ensure it's a 3-digit number



def convert_to_standard_date(date_str):
    # Parse the date
    year = int(date_str[:4])
    day_of_year = int(date_str[4:])

    # Convert to datetime object
    date_obj = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)  # Using day_of_year - 1 because timedelta is 0-indexed

    # Return in the desired format
    return date_obj.strftime('%Y%m%d')
