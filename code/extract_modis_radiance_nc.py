from extract_training_data import *

folder = "/uio/hume/student-u37/fslippe/for_Rob/"
start = "20200303"
end = "20230306"
start_converted = convert_to_day_of_year(start)
end_converted = convert_to_day_of_year(end)


x, dates, masks, lon_lats, times = extract_1km_data(folder,
                                                    bands=[1,2],
                                                    start_date=start_converted,
                                                    end_date=end_converted,
                                                    return_lon_lat=True,
                                                    data_type="hdf",
                                                    combine_pics=True)

create_nc_from_extracted_data(x,
                             dates,
                             masks,
                             lon_lats,
                             times,
                             save_folder=folder)





