from data_import import *
from data_analysis import *

def psatjira_data_analysis():
    plant = "PSATJIRA"
    ts_dict = import_precipitation_ts()
    precipitation_plots(ts_dict, plant)
    is_stationary(ts_dict, plant)

psatjira_data_analysis()