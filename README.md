# PDM-TMS Preprocessing pipeline

This repository contains python scripts used to preprocess TMS-EMG data collected for the PDM-TMS project in the Oliveira-Maia lab at the Champalimaud Foundation. 


**If you are doing a rotation in the lab, welcome!** 

Follow the steps bellow to ready your computer for TMS-EMG preprocessing and schedule a couple of hours with Francisco to preprocess your first TMS-EMG sessions!

## Setup

### Python installation and environment setup

#### Python installation
1. Install anaconda:

   Go to the following link and download anaconda: https://www.anaconda.com/products/individual#windows

2. Double click the installer to launch.

   Follow the installation steps, accepting all defaults

#### Environment setup
1. Copy the path to the environment file

   Right click the file and click "copy as path"

     _Example: "C:\Users\Admin\Downloads\TMS_preprocessing_env.yml"_

2. Go to search bar and enter:

   ```
   anaconda prompt
   ```   
   ![](/pipeline_setup/images/img1.png)

3. Type conda env create -f followed by the environment path (environment path must not have spaces):

     _Example:_
   ```
   conda env create -f C:\Users\Admin\Downloads\pdmTMS_preprocessing\pipeline_setup\TMS_preprocessing_env.yml
   ```
   ![](/pipeline_setup/images/img2.png)
   (this step may tak a few minutes)
   
   When finished installing, anaconda prompt will look like this:
   ![](/pipeline_setup/images/img3.png)

4. Close anaconda prompt

Environment setup done!

### Preparing spyderIDE
1. Go to search bar and enter:

   ```
   anaconda navigator
   ```

2. Open Anaconda Navigator

3. Select TMS preprocessing from the dropdown menu
   ![](/pipeline_setup/images/img4.png)

4. Launch Spyder
   ![](/pipeline_setup/images/img5.png)

5. Go to _Projects > New Project_ and click _Open Project..._
   ![](/pipeline_setup/images/img6.png)

6. Select _Existing Directory_,  navigate to the folder where you saved the scripts and click _Create_
   ![](/pipeline_setup/images/img7.png)

7. A side bar will be opened. Double click _Preprocessing.py_
   ![](/pipeline_setup/images/img8.png)
   ![](/pipeline_setup/images/img9.png)

8. Go to _Tools > Preferences_
   ![](/pipeline_setup/images/img10.png)

9.	Go to _IPython console > Graphics_ and select _Backend: Automatic_
   ![](/pipeline_setup/images/img11.png)

10.	Click _Apply_, then _OK_ and you are ready to preprocess the data!
