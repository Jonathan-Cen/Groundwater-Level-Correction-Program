import math
import pandas as pd
import time
import numpy as np
import warnings
import os
import datetime
start_time = time.time()
start_year = 1950
current = datetime.datetime.now()
current_year = current.year



def score_BM_1(benchmark_list_of_list):
    for i in range(len(benchmark_list_of_list)):
        distance = benchmark_list_of_list[i][1]
        record = benchmark_list_of_list[i][2]
        start = benchmark_list_of_list[i][3]
        if(start == 0):
            score = -9999999999999
        else:
            score = record * 10 - distance - (start - start_year)
        benchmark_list_of_list[i].append(score)
    return benchmark_list_of_list

def score_BM_2(benchmark_list_of_list):
    for i in range(len(benchmark_list_of_list)):
        benchmark_list_of_list[i].pop()
        distance = benchmark_list_of_list[i][1]
        record = benchmark_list_of_list[i][2]
        end = benchmark_list_of_list[i][4]
        if(end == 0):
            score = -9999999999999
        else:
            score = record * 10 - distance - (current_year - end)
        benchmark_list_of_list[i].append(score)
    return benchmark_list_of_list



def main():
    print "Step 2 is running..."
    print
    warnings.filterwarnings('ignore')


    if not os.path.exists("Pickle_Data"):
        os.makedirs("Pickle_Data")
    if not os.path.exists("Outcome_Data"):
        os.makedirs("Outcome_Data")
    ########################## Get User-Specified Distance ##############################
    print("This program searches for two nearest benchmarks for "+
          "groundwater wells within a user-specified distance in meters.")
    print
    print("For example, if you specify 100, i.e. 100 meters, this program will search "+
          "all the benchmarks within 100 meters of each well, and return UP TO 2 benchmarks.")
    print
    print
    critical_distance = float(raw_input("Please specify the distance and press 'Enter' to submit: "))

    
    ########################### Read Groundwater Well Locations ###################
    Well_GL_Locations = pd.read_pickle("Pickle_Data/Well_Locations_RL.pkl")
    ########################### Read All Benchmark Measurements ###################
    benchmark_measurements = pd.read_pickle("Pickle_Data/all_benchmarks_measurements.pkl")

    ########################### Read Groundwater Well Locations ###################
    benchmark_locations = pd.read_pickle("Pickle_Data/all_benchmark_locations.pkl")



    columns = ['Well', 'Benchmark_1','Benchmark_2', 'dist_1_[m]','dist_2_[m]','records_1', 'records_2']
    df = pd.DataFrame(columns = columns)
    print
    print("Based on your specifications: ")
    print



    no_record_list = []
    ####################### Pick All Benchmarks Within Distance ##################
    for i in range(len(Well_GL_Locations)):
        #--- Get location of each well ---#
        well_name = Well_GL_Locations.index[i]
        print("processing well " + str(well_name))
        well_loc_E = Well_GL_Locations['Location E [m]'].iloc[i]
        well_loc_N = Well_GL_Locations['Location N [m]'].iloc[i]
        output_list_of_list = []
        
        
        
        '''Get all benchmarks within the user-speified distance'''
        for j in range(len(benchmark_locations)):
            benchmark_name = benchmark_locations.index[j]
            benchmark_loc_E = benchmark_locations['Location E [m]'].iloc[j]
            benchmark_loc_N = benchmark_locations['Location N [m]'].iloc[j]
            distance_from_well = math.sqrt(math.pow((well_loc_E - benchmark_loc_E),2) + 
                                          math.pow((well_loc_N - benchmark_loc_N),2))
            if (distance_from_well <= critical_distance):
                output_list_of_list.append([str(benchmark_name),distance_from_well])
        
        '''Do not proceed if there is no benchmark available within the user-specified distance'''
        if(len(output_list_of_list) == 0):
            no_record_list.append(well_name)
            df.loc[i] = [well_name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            continue
        
        '''Get the further details of each benchmark that is within the distance'''
        for k in range(len(output_list_of_list)):
            benchmark_name_1 = output_list_of_list[k][0]
            benchmark_record = benchmark_measurements.loc[benchmark_name_1]
            benchmark_record.dropna(inplace = True) #drop the empty cells for each benchmark records
            num_record = len(benchmark_record)
            if (num_record == 0):
                start = 0
                end = 0
            else:
                start = benchmark_record.index[0]
                end = benchmark_record.index[-1]
            
            output_list_of_list[k].append(num_record)
            output_list_of_list[k].append(start)
            output_list_of_list[k].append(end)


        '''Select the most appropiate benchmarks'''
        for_BM_1 = score_BM_1(output_list_of_list)
        for_BM_1 = sorted(for_BM_1,key = lambda(name,dist,num,start,end,score):score)
        
        if(len(output_list_of_list) > 1):
            benchmark_1 = for_BM_1[-1][0]
            dist_1 = for_BM_1[-1][1]
            record_1 = for_BM_1[-1][2]
                
            for_BM_2 = score_BM_2(output_list_of_list)
            for_BM_2 = sorted(for_BM_2, key = lambda(name,dist,num,start,end,score):score)
            benchmark_2 = for_BM_2[-1][0]
            dist_2 = for_BM_2[-1][1]
            record_2 = for_BM_2[-1][2]
            
            if (benchmark_2 == benchmark_1):
                benchmark_2 = for_BM_2[-2][0]
                dist_2 = for_BM_2[-2][1]
                record_2 = for_BM_2[-2][2]
        else:
            benchmark_1 = for_BM_1[-1][0]
            dist_1 = for_BM_1[-1][1]
            record_1 = for_BM_1[-1][2]
            benchmark_2 = np.nan
            dist_2 = np.nan
            record_2 = np.nan        


            
        df.loc[i] = [well_name, benchmark_1, benchmark_2, dist_1, dist_2, record_1, record_2]
        


        
    df.set_index('Well', inplace = True)        
    df.to_excel('Outcome_Data/Outcome_Selected_Benchmarks_{}_meters.xlsx'.format(critical_distance))  

        
        
        
    delete_column = range(2,6)
    df.drop(df.columns[delete_column], axis = 1, inplace = True)
     
    df.to_pickle("Pickle_Data/Outcome_Selected_Benchmarks.pkl")
        
        

    end_time = time.time()
    print
    print("#"*94)
    print "-"*42 ,
    print "SUMMARY",
    print "-"*43
    print("#"*94)
    print
    print ("There are " + str(len(Well_GL_Locations)) + " wells inputed to this program, " + 
            str(len(no_record_list)) + " of them have no available benchmarks based on the specified distance.")
    print
    print("The following wells have no available benchmark within the specified distance: ")
    print([str(i) for i in no_record_list])
    print
    print('Exectution Time of Step 2: ' + str(round(end_time - start_time)) + " seconds")
    print"************************************* Step 2 is finished *************************************"
    print

if __name__ == '__main__':
    main()

