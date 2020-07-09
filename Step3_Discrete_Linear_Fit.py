import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
import datetime as dt
import time
import os
import sys
from pandas.plotting import register_matplotlib_converters
import bisect
import gc


current_year = dt.datetime.now().year

sys.setrecursionlimit(1500)





def generate_groundwater_level_graph(x,y,y2, plot_name):
    fig, ax = plt.subplots(figsize = (10,7))
    ax.plot(x,y)
    ax.scatter(x,y)
    ax.set_ylim(65,0)
    ax.set_title(plot_name + " Groundwater Level")
    ax.set_xlabel("year", fontsize = 14)
    ax.set_ylabel("Groundwater Level ~ wellhead (m)", color = 'red', fontsize = 14)
    ax.autoscale(True, axis = 'y')
    ax.set_yticks(np.arange(0, (max(y)/5) * 5 + 10, 5))
    
    ax2 = ax.twinx()
    ax2.plot(x, y2, color = 'blue')
    ax2.set_ylabel('Groundwater Level ~ Seawater (m)', color = 'blue', fontsize = 14)
    y2_start = int(y2.min()/5) * 5
    y2_scale = list(range(y2_start, y2_start + int(ax.get_yticks()[-1]) + 5, 5))
    ax2.set_yticks(y2_scale)

    filename = str(plot_name)
    if('/' in filename):
        filename = filename.replace('/','_')
    plt.savefig('Groundwater_Grapths/{}_GL.png'.format(filename))
    fig.clf()
    plt.close()
    del x, y, y2
    gc.collect()


def generate_subsidence_level_graph(x,y, benchmarks, plot_name):
    fig = plt.figure(figsize = (10,7))
    ax = fig.add_subplot(111)
    ax.scatter(x,y)
    ax.scatter(benchmarks['Datetime'], benchmarks['Benchmarks_Elevation'])#plot adjusted benchmarksx
    ax.set_xlabel("year")
    ax.set_ylabel("Elevation ~ Sealevel (m)")
    ax.set_title(plot_name + " Wellhead Level & Benchmark Level")
    ax.legend()
    filename = str(plot_name)
    if('/' in filename):
        filename = filename.replace('/','_')
    fig.savefig('Subsidence_Grapths/{}_Subsidence_Corrected.png'.format(filename))
    fig.clf()
    plt.close()
    del x, y, benchmarks
    gc.collect()



def plot_future(x,y, benchmarks, plot_name, future_predict):
    fig = plt.figure(figsize = (10,7))
    ax = fig.add_subplot(111)
    ax.scatter(x,y)
    ax.scatter(benchmarks['Datetime'], benchmarks['Benchmarks_Elevation'])#plot adjusted benchmarksx
    ax.set_xlabel("year")
    ax.set_ylabel("Elevation ~ Sealevel (m)")
    ax.set_title(plot_name + "  future prediction up to {}".format(future_predict))
    ax.legend()
    filename = str(plot_name)
    if('/' in filename):
        filename = filename.replace('/','_')
    fig.savefig('Future_Prediction_Linear/{}_future_predict_to_{}.png'.format(filename,future_predict))

    fig.clf()
    plt.close()
    del x, y, benchmarks
    gc.collect()


    
def linear_coefficient_and_intercept(x, y, BM_string_list):
    x = np.array(list(x)).reshape(-1,1)
    model = LinearRegression()
    try:
        model.fit(x,y)
    except ValueError,e:
        e = str(e)
        index = e.index(':')
        error = e[index+2: ]
        if(error not in BM_string_list):
            BM_string_list.append(error)
            return 'error','error', BM_string_list

    return np.float64(model.coef_), np.float64(model.intercept_), BM_string_list
    


def check_r_squared(input_DataFrame):
    model = LinearRegression()
    x = np.array(list(input_DataFrame.index)).reshape(-1,1)
    model.fit(x, input_DataFrame.values)
    return model.score(x, input_DataFrame.values)

def convert_datetime_to_float(dt):
    year = dt.year
    month = dt.month
    if(month == 1):
        days_in_previous_months = 0
    elif (month == 2):
        days_in_previous_months = 31
    elif (month == 3):
        days_in_previous_months = 59
    elif (month == 4):
        days_in_previous_months = 90
    elif (month == 5):
        days_in_previous_months = 120
    elif (month == 6):
        days_in_previous_months = 151
    elif (month == 7):
        days_in_previous_months = 181
    elif (month == 8):
        days_in_previous_months = 212
    elif (month == 9):
        days_in_previous_months = 243
    elif (month == 10):
        days_in_previous_months = 273
    elif (month == 11):
        days_in_previous_months = 304
    else:
        days_in_previous_months = 334
    day = dt.day
    total_days = days_in_previous_months + day
    return np.float64(year + np.float64(total_days) / 365.25)    

def convert_float_years_to_date(number):
    decimal = str(number-int(number))[1:]
    decimal = np.float64(decimal)
    year = np.int64(number)

    if(decimal <= np.float64(31)/365.25):
        month = 1
        day = np.int64(decimal * 365.25)
    elif(decimal > np.float64(31)/365.25 and decimal <= np.float64(59)/365.25):
        month = 2
        day = np.int64(decimal * 365.25 - 31)
    elif(decimal > np.float64(59)/365.25 and decimal <= np.float64(90)/365.25):
        month = 3
        day = np.int64(decimal * 365.25 - 59)
    elif(decimal > np.float64(90)/365.25 and decimal <= np.float64(120)/365.25):
        month = 4
        day = np.int64(decimal * 365.25 - 90)
    elif(decimal > np.float64(120)/365.25 and decimal <= np.float64(151)/365.25 ):
        month = 5
        day = np.int64(decimal * 365.25 - 120)
    elif(decimal > np.float64(151)/365.25 and decimal <= np.float64(181)/365.25):
        month = 6
        day = np.int64(decimal * 365.25 - 151)
    elif(decimal > np.float64(181)/365.25 and decimal <= np.float64(212)/365.25):
        month = 7
        day = np.int64(decimal * 365.25 - 181)
    elif(decimal > np.float64(212)/365.25 and decimal <= np.float64(243)/365.25):
        month = 8
        day = np.int64(decimal * 365.25 - 212)
    elif(decimal > np.float64(243)/365.25 and decimal <= np.float64(273)/365.25):
        month = 9
        day = np.int64(decimal * 365.25 - 243)
    elif(decimal > np.float64(273)/365.25 and decimal <= np.float64(304)/365.25):
        month = 10
        day = np.int64(decimal * 365.25 - 273)
    elif(decimal > np.float64(304)/365.25 and decimal <= np.float64(334)/365.25):
        month = 11
        day = np.int64(decimal * 365.25 - 304)
    else:
        month = 12
        day = np.int64(decimal * 365.25 - 334)
    if(day == 0):
        day = 1
    return dt.date(year, month, day)

def slope_intercept(x0,y0, x1, y1):
    slope = (y1 - y0)/(x1 - x0)
    intercept = y0 - slope * x0
    return slope, intercept

def linear_interpolation(x0, y0, x1, y1, x):
    return y0 + (x - x0)*((y1 - y0)/(x1 - x0))


def find_before_after(date_list, input_year): 
    last_year = date_list[-1]
    bisect.insort(date_list, input_year)
    before = date_list.index(input_year) - 1
    after = date_list.index(input_year)
    date_list.remove(input_year)
    if(input_year < last_year):
        return before, after
    else:
        return before, before

def discrete_linear_fit(dates_list, depth_list,x):
    before_index, after_index = find_before_after(dates_list, x)
    if(before_index != after_index):
        x0 = dates_list[before_index]
        y0 = depth_list[before_index]
        
        x1 = dates_list[after_index]
        y1 = depth_list[after_index]
        return linear_interpolation(x0, y0, x1, y1, x)
    else:
        x0 = dates_list[-2]
        y0 = depth_list[-2]
        
        x1 = dates_list[-1]
        y1 = depth_list[-1]
        slope, intercept = slope_intercept(x0, y0, x1, y1)
        return slope * x + intercept

    
def main():
    if not os.path.exists('Groundwater_Grapths'):
        os.makedirs('Groundwater_Grapths')
    if not os.path.exists('Subsidence_Grapths'):
        os.makedirs('Subsidence_Grapths')
    if not os.path.exists("Outcome_Data"):
        os.makedirs("Outcome_Data")
    print("Step 3 is running...")
    print
    print "This program does the discrete linear fit of benchmark data and the groundwater level correction."
    print
    future_predict = int(raw_input("Enter future prediction year, press 'Enter' to submit.\n(If you do not wish to do any future prediction, enter a number that is prior to current year) \nEnter your number: "))
    print
    pd.set_option('mode.chained_assignment', None)
    pd.options.display.max_rows = 100

    start_time = time.time()
       
    register_matplotlib_converters() #Suppress warning

    BM_string_list = []

    plt.rcParams.update({'figure.max_open_warning': 0})
    style.use('ggplot')
    
    writer1 = pd.ExcelWriter('Outcome_Data/Outcome_Subsidence_Correction.xlsx',engine='xlsxwriter')
    writer2 = pd.ExcelWriter('Outcome_Data/Outcome_Adjusted_Benchmarks.xlsx',engine='xlsxwriter')
    if(future_predict >= current_year):
        writer3 = pd.ExcelWriter('Outcome_Data/Outcome_Future_Prediction_Linear.xlsx',engine = 'xlsxwriter')
        if not os.path.exists('Future_Prediction_Linear'):
            os.makedirs('Future_Prediction_Linear')

     
        
        
    ######################################################################################################
    ################################## Read benchmark of wells ###########################################
    ######################################################################################################


    data = pd.read_pickle('Pickle_Data/Outcome_Selected_Benchmarks.pkl') #set the first column to as index

    data.fillna(0, inplace = True) #Turn the NaN to 0


    ######################################################################################################
    ####################################### Read benchmarks ##############################################
    ######################################################################################################

    benchmark_measurements = pd.read_pickle('Pickle_Data/all_benchmarks_measurements.pkl')


    ######################################################################################################
    #################################### Read Initial Elevation ##########################################
    ######################################################################################################


    initial_survey = pd.read_pickle('Pickle_Data/Well_Locations_RL.pkl')
    delete_columns = [2,3]
    initial_survey.drop(initial_survey.columns[delete_columns], axis = 1, inplace = True)



    ######################################################################################################
    ##################################### Read Historical GL #########################################
    ######################################################################################################
    Historical_GL = pd.read_pickle('Pickle_Data/Historical_Groundwater_Level_data.pkl')

    ######################################################################################################
    ##################################### Handle Exceptions ##############################################
    ######################################################################################################
    no_historical_GL = []
    no_benchmark_available = []
    na = []
    ######################################################################################################
    ################################# The program starts here ############################################
    ######################################################################################################

    for i in range(len(data)):#len(data)
        plot_name = data.index[i]
        print ("processing well " + str(plot_name))
        if(data.iloc[i][0] == 0):
            na.append(plot_name)
            continue
        

        #create a copy of survey data for a particular year
        one_survey = initial_survey.iloc[i].to_frame().T
        initial_date = one_survey.iloc[0][0]
        initial_elevation = np.float64(one_survey.iloc[0][1])
        initial_date = convert_datetime_to_float(initial_date)
        one_survey.drop('Well Date',1, inplace = True)
        one_survey.insert(0,'Well Date',initial_date,True)

        s1 = benchmark_measurements.loc[data.iloc[i][0]].dropna()
        s1 = s1[s1.index > initial_date] #not to use benchmark data that's prior to well existence
        if(len(s1)<=0):
            no_benchmark_available.append(plot_name)
            continue
      
        
        
        if (data.iloc[i][1] != 0):  #s2 could be empty, and hence would be converted to 0
            s2 = benchmark_measurements.loc[data.iloc[i][1]].dropna()

            s2 = s2[s2.index > initial_date]  #not to use benchmark data that's prior to well existence 
            
            #---s1 linear fit---
            s1_coefficient, s1_intercept, BM_string_list = linear_coefficient_and_intercept(s1.index, s1.values, BM_string_list)
            if(s1_coefficient == 'error'):
                continue
            #---Adjust offset for benchmark 1 against initial survey---
            survey_pred = initial_date * s1_coefficient + s1_intercept
            offset_1 = abs(survey_pred - initial_elevation)
            if(survey_pred > initial_elevation):
                for j in range(len(s1.values)):
                    s1.values[j] -= np.float64(offset_1)  #s1_adjusted
            else:
                for j in range(len(s1.values)):
                    s1.values[j] += np.float64(offset_1)


            #---s1 adjusted fit---
            #concatinate initial and s1 two series to a dataframe
            
            survey_data = one_survey.iloc[0].values
            df = pd.DataFrame(survey_data[[1]], index = survey_data[[0]])
            initial_s1_combined = pd.concat([df, s1])
            if(len(s2) > 0):

                #---s2 linear fit---
                s2_coefficient, s2_intercept, BM_string_list = linear_coefficient_and_intercept(s2.index, s2.values, BM_string_list)
                if(s2_coefficient == 'error'):
                    continue
                #---Adjust Offset for benchmark 2 against initial_s1_combined---
        
                initial_s1_combined_coefficient, initial_s1_combined_intercept, BM_string_list = linear_coefficient_and_intercept(initial_s1_combined.index,
                                                                                                                  initial_s1_combined.values,
                                                                                                                  BM_string_list)
        
                mid_time = ((s2.index[0] - initial_s1_combined.index[-1])/2) + initial_s1_combined.index[-1]
        
        
                initial_s1_combined_pred_mid_time = mid_time * initial_s1_combined_coefficient + initial_s1_combined_intercept
                s2_pred_mid_time = mid_time * s2_coefficient + s2_intercept
        
                #---Find offset---
                offset_2 = abs(s2_pred_mid_time - initial_s1_combined_pred_mid_time)
        
                if(s2_pred_mid_time > initial_s1_combined_pred_mid_time):
                    for j in range(len(s2.values)):
                        s2.values[j] -= np.float64(offset_2)
                else:
                    for j in range(len(s2.values)):
                        s2.values[j] += np.float64(offset_2)
                
                all_combined = pd.concat([initial_s1_combined,s2])
            else:
                all_combined = initial_s1_combined


        else: #Only one benchmark
            #---s1 linear fit---
            s1_coefficient, s1_intercept, BM_string_list = linear_coefficient_and_intercept(s1.index, s1.values, BM_string_list)
            if(s1_coefficient == 'error'):
                continue
            #---Adjust offset for benchmark 1 against initial survey---
            survey_pred = initial_date * s1_coefficient + s1_intercept
            offset_1 = abs(survey_pred - initial_elevation)

            if(survey_pred > initial_elevation):
                for j in range(len(s1.values)):
                    s1.values[j] -= np.float64(offset_1)  #s1_adjusted
            else:
                for j in range(len(s1.values)):
                    s1.values[j] += np.float64(offset_1)

            #---s1 adjusted fit---
            #concatinate initial and s1 two series
            survey_data = one_survey.iloc[0].values
            df = pd.DataFrame(survey_data[[1]], index = survey_data[[0]])
            all_combined = pd.concat([df, s1])


    ######################################################################################################


        all_combined.reset_index(inplace = True)
        all_combined.rename(columns = {'index':'Date', 0:'Benchmarks_Elevation'}, inplace = True)
        all_combined['Datetime'] = all_combined.apply(lambda row: convert_float_years_to_date(row.Date),
                                                      axis = 1)

        all_combined.drop_duplicates(subset = 'Date', keep='first', inplace = True)
        all_combined.sort_values(by = ['Date'], inplace = True)


        dates_list = all_combined['Date'].tolist()
        depth_list = all_combined['Benchmarks_Elevation'].tolist()

    
        all_combined.set_index('Date', inplace = True)
        filename = str(plot_name)
        if('/' in filename):
            filename = filename.replace('/','_')
        all_combined.to_excel(writer2, sheet_name = filename)

        ##################################################################################################
        ################################### Extract Historical GL Data ###################################
        ##################################################################################################
        

        try:
            dfs = Historical_GL.groupby(Historical_GL.index).get_group(plot_name)
        except KeyError:
            no_historical_GL.append(plot_name)
            continue
        
        df = dfs[dfs.Depth != 0] #drop the 0 values
        df.rename(columns = {'Measurement Date':'Measurement_Date'}, inplace = True) #rename column
        if (len(df['Measurement_Date']) == 0):
            continue
        
        df['Date'] = df.apply(lambda row: convert_datetime_to_float(row.Measurement_Date), axis = 1)
        df = df[['Measurement_Date','Date','Depth']]
        
        df['Adjusted_GW_level'] = df.apply(lambda row: np.float64(discrete_linear_fit(dates_list, depth_list, row.Date)- row.Depth)\
          if row.Date > initial_date else np.float64(initial_elevation - row.Depth), axis = 1)
        
        df['Wellhead_Level'] = df.apply(lambda row: np.float64(discrete_linear_fit(dates_list, depth_list, row.Date))\
          if row.Date > initial_date else np.float64(initial_elevation), axis = 1)
                                             
        #---Groundwater Level Graph---#
        generate_groundwater_level_graph(df['Measurement_Date'],df['Depth'],df['Adjusted_GW_level'], str(plot_name)) #x, y, y2, plot_name

        #---Subsidence Level Graphs---#
        generate_subsidence_level_graph(df['Measurement_Date'], df['Wellhead_Level'], all_combined, str(plot_name)) #x, y, all_benchmarks, plot_name

    ######################################################################################################
    ########################### Future prediction for acceptable linear fit ##############################
    ######################################################################################################

        if (future_predict >= current_year):
            df2 = df['Date']
            future_years = np.arange(df2.iloc[-1] + 0.5, future_predict+0.5, 0.5)
            df2 = df2.append(pd.Series(future_years),ignore_index = False)

            df2 = df2.to_frame(name = 'Date_Predict')
            df2['Datetime_Predict'] = df2.apply(lambda row: convert_float_years_to_date(row.Date_Predict), axis = 1)
            x0 = dates_list[-2]
            y0 = depth_list[-2]
            
            x1 = dates_list[-1]
            y1 = depth_list[-1]
            
            slope, intercept = slope_intercept(x0, y0, x1, y1)
            
            df2['Wellhead_Level_Linear_predict'] = df2.apply(lambda row: np.float64(initial_elevation) if row.Date_Predict <= initial_date\
               else (np.float64(discrete_linear_fit(dates_list, depth_list, row.Date_Predict)) if row.Date_Predict <= dates_list[-1] \
                     else np.float64(row.Date_Predict * slope + intercept) ),
                     axis = 1)


            plot_future(df2['Datetime_Predict'], df2['Wellhead_Level_Linear_predict'],all_combined, str(plot_name),future_predict)
            filename = str(df.index[0])
            if('/' in filename):
                filename = filename.replace('/','_')
            df2.to_excel(writer3, sheet_name = filename)

    ######################################################################################################
    ##################################### Saving Output Files ############################################
    ######################################################################################################
        

        filename = str(df.index[0])
        if('/' in filename):
            filename = filename.replace('/','_')
        df.to_excel(writer1, sheet_name = filename)
        
        
    writer1.save()
    writer2.save()
    if(future_predict >= current_year):
        writer3.save()
    print
    print("#"*94)
    print "-"*42,
    print "SUMMARY",
    print "-"*43
    print("#"*94)
    print
    print "------Quick Summary------"
    print "Total number of input: %d"%len(data)
    print "%d wells with no benchmark available."%len(na)
    print "%d wells with no historical groundwater level data."%len(no_historical_GL)
    print "%d wells have no benchmark records available after initial survey."%len(no_benchmark_available)

    print "%d comment(s) found in all_benchmarks.xlsx"%len(BM_string_list)
    print "Total number of successful output: %d"%(len(data) - len(na) - len(no_historical_GL)
                                                   - len(no_benchmark_available) - len(BM_string_list))
    print
    print "------Detailed Summary-----"
    print
    print("Wells with no benchmark available based on the results of Step 2:")
    print([str(i) for i in na])
    print
    print("Wells with no historical groundwater level data: ")
    print([str(i) for i in no_historical_GL])
    print
    if(len(no_benchmark_available)!=0):
        print("Wells have no benchmark records available after initial survey:")
        print([str(i) for i in no_benchmark_available])
    print
    if(len(BM_string_list) != 0):
        print("The following comments exist in the all_benchmarks.xlsx file. Please replace them with np.nan in Step 1 and run Step 1 again")
        print (BM_string_list)
    print
    end_time = time.time()
    print("Execution Time of Step 3: " + str(round(end_time - start_time)) + " seconds")

    
    print"************************************* Step 3 is finished *************************************"
    print



if __name__ == '__main__':
    main()





























