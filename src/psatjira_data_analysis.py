from data_import import import_precipitation_ts
from data_analysis import precipitation_plots, is_stationary


def psatjira_data_analysis():
    plant = "PSATJIRA"
    ts_dict = import_precipitation_ts()
    precipitation_plots(ts_dict, plant)
    is_stationary(ts_dict, plant)


if __name__ == "__main__":
    psatjira_data_analysis()
