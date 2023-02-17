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