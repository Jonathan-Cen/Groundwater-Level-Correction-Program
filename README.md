# Groundwater_Level_Correction_Program

## Summary
Monitoring geothermal groundwater level will provide useful information about the geothermal reservoir. A common way to measure geothermal groundwater level is by measuring the distance between the groundwater-surface and the wellhead. However, as reservoir pressure changes, subsidence might occur at a particular area, lower the wellhead level, and result in an inaccurate groundwater level measurement. This program is designed to automatically correct the groundwater level of a large number of groundwater wells by referencing the groundwater level to the sea level. Nearby benchmarks’ elevations will be used as representative of the wellhead elevation in the calculation after offset adjustments.

## Python Version: Python 2.7
**Modules Required:**
- Tkinter
- Numpy
- Pandas
- Matplotlib
- Scikit-Learn
- Bisect

## Input Data
This program takes three Excel spreadsheets as input data:
-	All_Benchmarks.xlsx
-	Historical_Groundwater_Level_Data.xlsx
-	Well_Locations_RL.xlsx

### Templates of all these input data are included in the repository inside the "input" folder.

**All_Benchmarks.xlsx contains:**
-	Benchmark labels
-	Benchmark coordinates. (Please use a consistent coordinate system for all input data, for example the coordinate system used in this program is NZMG, but it should work for other coordinate systems)
-	Benchmark levels over time

**Historical_Groundwater_Level_Data.xlsx contains:**
-	Well labels
-	The survey date of groundwater level
-	The depth (measured in meters), i.e. the historical groundwater level measured from the wellhead.

**Well_Locations_RL.xlsx contains:**
-	Well labels
-	Well Date (the date the well was drilled)
-	The Reduced Level (the elevation of the wellhead with reference to sea level)
-	The coordinates of the well

## Output Data
-	A Groundwater_Graphs folder that contains the corrected groundwater level over time for all wells
-	A Subsidence_Graphs folder that contains the corrected wellhead level over time for all wells.
-	A Future_Prediction_Linear folder that contains the predicted wellhead level for all wells. (This folder only exists if you choose to perform a future prediction)
-	An Outcome_Data folder that contains all output data in Excel spreadsheet format.



## Instructions
1.	Fill in your data in the input spreadsheets following the templates given in the **input** folder
2.	Double click Run_Program.py to launch the program
3.	When the program is launched, you should see a command window and a control panel as follows:

**Command Window:**
  ![Alt text](/screenshots/command_window.png?raw=true "Command Window")

**Control Panel:**

![Alt text](/screenshots/control_panel.png?raw=true "Control Panel")

4.	Click to run a step. Please note, if this your first time running the program, please run the steps in order. A s ummary is provided after the execution of each step.
5.	Run Step 1:
  ![Alt text](/screenshots/step1.png?raw=true "Step 1")
6.	Run Step 2:
    - In the command prompt, you will need to specify a distance. The program will seek for up to 2 benchmarks for each well within the distance you specified (in this example, I specified 200. i.e. 200 meters) Press enter to submit:
    ![Alt text](/screenshots/step2_1.png?raw=true "Step 2 - 1")
    - Step 2 is running:
    
    ![Alt text](/screenshots/step2_2.png?raw=true "Step 2 - 2")
7.	Run Step 3 (You will need to specify a future year that you want to predict for the wellhead level. If you do not wish to do any future prediction, enter a year that is prior to the current year.In this example, I wish to predict the wellhead level up to 2025): 
![Alt text](/screenshots/step3.png?raw=true "Step 3")
8.	General recommendation: If you decide to enter a different distance for Step 2, please delete all the output files produced by the previous run. Because a new set of benchmarks will result in a different number of wells that are eligible for groundwater level correction, previous valid outputs might be invalid for the current criteria.


