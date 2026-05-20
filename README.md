# Gatorade AKA "Water Drops" AKA "Dropping Drops" AKA "Droping Drops"
An exhibit where water valves draw a picture by opening and closing their valves over time. see [this](https://www.youtube.com/watch?v=FG_l1oacWoQ) video for a demo.

see [full documentation here](https://madaorgil.sharepoint.com/:f:/s/MakeMada/IgAcPqMIf3aWQ6yXt_N7ErjtAXLbkVMVVD-4b84fQm6kC6U?e=27vzwn)

# Current Code State

- the code that's running in the exhibit is:
    - the `currently_running_version` in the `code python` folder
    - the `currently_running_version` folder in the `code arduino` folder

Then new version  in the camera is in `src` of `code python` and in `Drop-Screen` in `code arduino` with much faster camera fps and responsivity,  mainly because we removed the delays in the .ino code and by using `cv2` without `pygame`.  
It has the following problems:
1. some miscommunication causes some casettes to spontaneously drop some square of water, or sometimes even a huge blob of water. happens less when fps is 1, more when it is 0.33.

# Requirements:
- Arduino nano
- Python machine (rpi or pc)
- camera connected to python machine via USB
- arduino connected to python via USB
# Installation
1. for arduino: burn the arduino code onto an arduino nano (`code arduino/Drop-Screen`)
2. for python: create venv with requirements.txt and run main.py.  
    2.1. on Raspberrypi, you can run this which will set everything up for you (including autostart of the app, anydesk and making the mouse disappear):
```bash
cd "code python\src"
chmod +x setup_gui_rpi.sh && ./setup_gui_rpi.sh
./run.sh
```