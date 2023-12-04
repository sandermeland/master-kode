from dataclasses import dataclass


@dataclass
class TimeFrame:
    year : int
    start_month : int 
    end_month : int 
    start_day : int 
    end_day : int 
    start_hour : int 
    end_hour : int
    


one_hour = TimeFrame(year = 2023, start_month = 6, end_month = 6, start_day = 26, end_day = 26, start_hour = 15, end_hour = 16) # it may be possible to start from hour 14

one_day = TimeFrame(year = 2023, start_month = 6, end_month = 6, start_day = 26, end_day = 26, start_hour = 0, end_hour = 23)

one_week = TimeFrame(year = 2023, start_month = 6, end_month = 6, start_day = 19, end_day = 25, start_hour = 0, end_hour = 23)

half_month =  TimeFrame(year = 2023, start_month = 6, end_month = 6, start_day = 14, end_day = 30, start_hour = 0, end_hour = 23)

one_month = TimeFrame(year = 2023, start_month = 6, end_month = 6, start_day = 1, end_day = 30, start_hour = 0, end_hour = 23)
