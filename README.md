# `qgis_style`

Python code for generating QGIS style files.

## Usage

```console
python3 make_style_files.py input.txt
```

The input file tells the function what to do. It is a text file where each line is an argument name followed by argument values. The order of the lines does not matter. Every input file must define the argument `task`, for example

```
task make_linear_colour_ramp
``` 

 Various examples are shown below.

### Make a linear colour ramp

In this case, the input file looks like

```
task make_linear_colour_ramp
n_values 11 
rgb_min 0.0 0.0 0.0
rgb_max 1.0 1.0 1.0
dir_out output/
ramp_name black_to_white
```

### Make filled contours

The input file should be something like:

```
task make_filled_contours
z_min 2250.0
d_z 250.0 
n_bins 7 
colour_ramp_type pyplot_linear 
pyplot_ramp_name magma
outline_contrast 0.15
outline_style solid
outline_width 2.0
outline_width_unit Pixel 
dir_out test/output/
qgis_version 3.22.9-Białowieża
style_file_name filled_contours
```

