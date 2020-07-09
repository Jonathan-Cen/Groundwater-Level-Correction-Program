#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
############################ Do Not Modify ##########################################
import numpy as np
import pandas as pd
import time
import os
def main():
    print"Step 1 is running..."
    print
    print "This program does the data cleaning for input data. You might not see anything appear on this window for a while."
    pd.set_option('display.max_columns', 20)
    start_time = time.time()
    if not os.path.exists("Pickle_Data"):
        os.makedirs("Pickle_Data")
    if not os.path.exists("Outcome_Data"):
        os.makedirs("Outcome_Data")
        

    ################## Read benchmarks Values ##############

    excel_file = 'All_Benchmarks.xlsx'
    benchmark_locations = pd.read_excel(excel_file, index_col = 0)   
    benchmark_measurements = benchmark_locations.drop(benchmark_locations.columns[0:2], axis = 1, inplace = False)

    #################################################################################################
    ####### Add a new entry below if you encounter any comment in the all_benchmarks.xlsx file ######
    #################################################################################################
    benchmark_measurements.replace({'not found':np.nan, 'dest.': np.nan, 'destroyed':np.nan, 'no access':np.nan,
                                    'found May-04':np.nan, 'under lid':np.nan, "Dest":np.nan,
                                    "under rock?":np.nan, "now AA23/2":np.nan, 'under pipe':np.nan, 
                                    'My Comment':np.nan}, inplace = True)

    benchmark_measurements.to_pickle('Pickle_Data/all_benchmarks_measurements.pkl')
    print
    print("#"*94)
    print "-"*42 ,
    print "SUMMARY",
    print "-"*43
    print("#"*94)
    print
    print('all_benchmarks_measurements is saved as pickle')
    print
    
    ################## Read benchmarks Locations ###########
    
    benchmark_locations.drop(benchmark_locations.columns[2:], axis = 1, inplace = True)
    
    
    benchmark_locations.to_pickle("Pickle_Data/all_benchmark_locations.pkl")
    print("all_benchmark_locations is saved as pickle")
    print

    

    

    ############## Read RL & Locations ##############

    excel_file = "Well_Locations_RL.xlsx"
    Location_RL = pd.read_excel(excel_file)
    Location_RL = Location_RL.drop_duplicates(subset = "Well", keep = "first").set_index("Well")
    Location_RL.to_excel('Outcome_Data/Outcome_GL_Locations_RL.xlsx')
    Location_RL.to_pickle("Pickle_Data/Well_Locations_RL.pkl")
    print(excel_file + " is saved as pickle")
    print


    ############## Read Historical GW level #########

    excel_file = "Historical_Groundwater_Level_Data.xlsx"
    GL = pd.read_excel(excel_file, index_col = 0)
    GL.to_pickle('Pickle_Data/Historical_Groundwater_Level_data.pkl')
    print(excel_file + ' is saved as pickle')
    print


    end_time = time.time()
    print("Execution Time of Step 1: " + str(round(end_time - start_time)) + " seconds")
    print"************************************* Step 1 is finished *************************************"
    print

if __name__ == "__main__":
    main()
