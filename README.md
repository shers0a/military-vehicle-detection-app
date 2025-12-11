# military-vehicle-detection-app
This is the app i used for ROSPIN SATELLITE DATA PROCESSING MASTERCLASS 2025    
It works on RGB satellite imagery, using YOLOv8, and it has 2 purposes, 2 options in which it can be used:
1. Static detection, which consists in uploading a file with the maximum capacity 2000 MB and it detects the vehicles
  in it, through a slicing procedure with the SAHI library. It also counts the vehicles and sorts them in different classes:

    i1. Small_Military_Vehicle   
    i2: Large_Military_Vehicle  
    i3: Armored_Fighting_Vehicle  
    i4: Civilian_Vehicle   
    i5: Military_Construction_Vehicle

3. Dynamic detection: uploading 2 photos of the same region in 2 different period of times; T0 and T1.
   The app will return the density of vehicles per hectare and a heatmap, a matrix that will show on each grid
  (50m*50m sectors) how many vehicles left.
  Interpretation: (-) = vehicles left
                  (+) = vehicles came

The app works through streamlit and it oppens through the command: "streamlit run app.py" in the terminal.
!!IMPORTANT!! 
Works only on python 3.11, for which i created the environment. all the libraries needed to run the program are:
ultralytics, pytorch( torch visual, torch audio ), sahi, matplotlib, PIL, seaborn, numpy.

app.py = the app code (front end)
antrenare.py = the settings used for training the yolo model. use the file to train your own with the command "python antrenare.py" in the terminal
config.yaml = the configuration file used for the paths and classes
hmp.py = the code for the heatmap, apart from the one from app.py

This is the structure you should use for the project or create your own folder:                        
main folder(name)---> dataset---> train and val--> images file and labels file for each                

ENJOY!
