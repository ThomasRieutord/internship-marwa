# internship-marwa
Repository for keeping track and sharing code during Marwa's internship

The internship topic
--------------------
Data-driven forecasting consists in using AI to perform weather forecasts solely based on archived weather data. While its performance is quickly increasing, the current representation of the land cover in such forecasting is limited to a land/sea mask and ground elevation. However, the land cover (tree, grass, lake, town...) has a notorious importance on local weather. The internship will explore different ways to integrate the information contained in land cover maps into a data-driven forecast and assess the observed impact. Data will be provided to the student in order to start experimenting on small offline examples. The first problem will be to train an appropriate neural network to derive the 2-m temperature from the 30-m temperature and the land cover, over Ireland.  The chosen neural network must correctly deal with the various nature of the variables (categorical or continuous). Then the problem will be complexified by adding other variables, such as wind, humidity, elevation etc.


Getting started
---------------

### Installations
During this internship, the main coding language is Python. We will rely on the following packages:
```
numpy
xarray
scikit-learn
pytorch
pytorch-lightning
matplotlib
cartopy
```
We start from the [Anaconda](https://www.anaconda.com) distribution and install all required packages in an environment.

Additionally, we will use [metplotlib](https://github.com/ThomasRieutord/metplotlib) for the plotting for meteorological fields.
Please follow the installation guidelines in there.


### Get the data
The data is a file NetCDF file (2GB), that can be dowloaded at this [link](https://drive.proton.me/urls/C419HAAY8C#ZCWD89hawMtP).
It must be stored in the `data` repository (untracked by git).


### Get the code
Fork the present repository and clone it on your machine.


### Start your own code
To start coding on your own, create a new program (with an explicit name), e.g. `pytorch_intro_tutorial.py` in the current directory.
You can import your own functions from there with a simple `import myfunctions` at the begining of your program.
Edit the file `myfunctions.py` with your new functions as needed.
Once your program is working and clean, you can add it to the Git tracking and share it.


Guidelines for coding
---------------------

We use the following conventions to keep the code as clean as possible:
  * For indentation, use spaces, avoid tabs (only tolerated to ensure consistency)
  * Class names use camel case (`ThisIsCamelCase`), function names and variables names use snake case (`this_is_snake_case`), global variables are upper case.
  * The `myfunctions.py` file must contain exclusively functions. No main code nor data are allowed. Main codes (files that can be run using `python file.py`) are meant to be stored in this directory with an explicit mame (e.g. `example_of_accessing_and_plotting_data.py`. Symmetrically, no importable functions or classes should be defined in a main code.
  * Formatting code with [Black](https://black.readthedocs.io/en/stable/index.html) and [Isort](https://pypi.org/project/isort/) is mandatory.
