from Tkinter import *
import Step1_Data_Cleaning
import Step2_Determine_Benchmarks_for_Wells
import Step3_Discrete_Linear_Fit

def close_window():
    window.destroy()
def run_step_1():
    Step1_Data_Cleaning.main()
def run_step_2():
    Step2_Determine_Benchmarks_for_Wells.main()
def run_step_3():
    Step3_Discrete_Linear_Fit.main()



window = Tk()

# while(True):
print("Welcome to Groundwater Level Correction Program Command Window\n\n")
window.title("Groundwater Correction Program")
window.configure(width = 1000, height = 500, background = "black")
##photo = PhotoImage(file="well_image.gif")
##Label(window,image = photo, bg = "black").grid(row = 0, column = 0)

Label(window, text = "Welcome to Groundwater Level Correction Program", bg ="black",
     fg = "white", font = "none 20 bold").grid(row = 0, column = 0)
Label(window, text = "\nPlease click which step you want to run:\n", bg = 'black', fg = "white", font = "none 15 bold").grid(row = 1, column = 0)

Button(window, text = "Run Step 1", font = "bold", width = 10, height = 5, command = run_step_1).grid(row = 7, column = 0, sticky = W)
Button(window, text = "Run Step 2", font = "bold", width = 10, height = 5, command = run_step_2).grid(row = 7, column = 0)
Button(window, text = "Run Step 3", font = "bold", width = 10, height = 5, command = run_step_3).grid(row = 7, column = 0, sticky = E)


Label(window, text="\n\nClick to exit program:\n",bg = "black",fg ="white", font = "none 12 bold").grid(row= 8, column = 0)
Button(window, text = "Exit", width = 14, command = close_window).grid(row=9, column = 0)
window.mainloop()
window.mainloop()