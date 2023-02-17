'''
qgis_style/make_style_files.py
See qgis_style/README.md for usage.
'''

# --- Import modules. ---------------------------------------------------------

# Import standard-library modules.
import argparse
import os
import xml.etree.ElementTree as ET

# Import third-party modules.
import matplotlib
import numpy as np

# --- Handling XML. -----------------------------------------------------------
def set_xml_attributes(element, dict_):
    '''
    Wrapper for simple for-loop setting XML element key-value pairs from dictionary.
    '''

    for key, value in dict_.items():

        element.set(key, value)

    return

def write_xml_file(main_element, header_lines, path_out):
    '''
    Write an XML ET object to specified output path, with given header lines.
    '''
    
    # Write XML file.
    ET.indent(main_element)
    xml_string = ET.tostring(main_element, encoding = 'utf-8').decode('utf-8')
    print("Writing to {:}".format(path_out))
    with open(path_out, 'w', encoding = 'utf-8') as out_id:
        
        # Write each header line.
        for header_line in header_lines:
                
            # Use str.encode() to convert the header  string to bytes.
            out_id.write((header_line + '\n'))

        # Write the XML.
        out_id.write(xml_string)

    return

def write_color_ramp_to_qgis_xml(path_out, color_ramp_name, ramp_points, rgb_values, a_values):
    '''
    Writes a colour ramp as a QGIS XML style file.
    
    Input:

    path_out        Where colour ramp XML file will be written to.
    color_ramp_name The name of the colour ramp.
    ramp_points     1-D float array of length (n_edges), starting at 0.0 and
                    ending at 1.0, specifying the points along the ramp at
                    which the colours are specified. Must be sorted.
    rgb_values      3-D float array of shape (n_edges, 3) specifying the
                    RGB triples normalised from 0.0 to 1.0 (will be converted
                    to integers here).
    a_values        1-D float array of shape (n_edgers) specifying the alpha
                    values normalised from 0.0 to 1.0 (will be converted to
                    integers here).

    Notes:

    Using the Python XML library:
    https://stackabuse.com/reading-and-writing-xml-files-in-python/

    How color ramps are formatted in XML for QGIS:
    https://gis.stackexchange.com/a/269786/95769
    <!DOCTYPE qgis_style>
    <qgis_style version="1">
      <symbols/>
      <colorramps>
        <colorramp type="gradient" name="Custom_blues">
          <prop k="color1" v="247,251,255,255"/>
          <prop k="color2" v="8,48,107,255"/>
          <prop k="continuos" v="0"/>
          <prop k="stops" v="0.115;198,219,239,255:0.326;222,235,247,255:0.651;66,146,198,255:0.902;8,81,156,255"/>
        </colorramp>
      </colorramps>
    </qgis_style>
    '''

    # Check input values.
    n_edges = len(ramp_points)
    tol = 1.0E-12
    assert np.abs(ramp_points[0]) < tol
    assert np.abs(ramp_points[-1] - 1.0) < tol
    assert np.all(np.diff(ramp_points) > tol)
    #
    assert len(rgb_values.shape) == 2
    assert rgb_values.shape[0] == n_edges
    assert rgb_values.shape[1] == 3
    assert np.all(rgb_values >= -tol)
    assert np.all(rgb_values <= (1.0 + tol))
    #
    assert len(a_values.shape) == 1 
    assert a_values.shape[0] == n_edges 
    assert np.all(a_values >= -tol)
    assert np.all(a_values <= (1.0 + tol))

    # Convert floating values to integer.
    rgb_values = rgb_float_to_int(rgb_values)
    a_values   = rgb_float_to_int(a_values)

    # Create the file structure.
    qgis_style = ET.Element('qgis_style')
    qgis_style.set('version', '1')
    symbols = ET.SubElement(qgis_style, 'symbols')
    colorramps = ET.SubElement(qgis_style, 'colorramps')
    colorramp = ET.SubElement(colorramps, 'colorramp')
    colorramp.set('type', 'gradient')
    colorramp.set('name', color_ramp_name)
    #
    color12_fmt = '{:>d},{:>d},{:>d},{:>d}'
    color_ramp_color_1_str = color12_fmt.format(*rgb_values[0, :], a_values[0])
    color_ramp_color_2_str = color12_fmt.format(*rgb_values[-1, :], a_values[-1])

    # Format the 'stops' (ranges within which each colour is used).
    stops_str = ''
    for i in range(1, n_edges - 1):

        stops_str_i = '{:>5.3};{:>d},{:>d},{:>d},{:>d}'.format(
            ramp_points[i], *rgb_values[i, :], a_values[i])

        stops_str = stops_str + stops_str_i

        if i != (n_edges - 2):

            stops_str = stops_str + ':'

    # Store colour ramp variables in dictionary.
    colorramp_prop_dict = {
            'color1'    : color_ramp_color_1_str,
            'color2'    : color_ramp_color_2_str,
            'stops'     : stops_str }

    # Store the colour ramp variables in a subelement of the colorramp
    # XML ET object.
    for color_ramp_key, color_ramp_value in colorramp_prop_dict.items():

        prop = ET.SubElement(colorramp, 'prop')
        prop.set('k', color_ramp_key)
        prop.set('v', color_ramp_value)

    # Add some header lines (seems to not be supported by Python XML).
    header_lines = ['<!DOCTYPE qgis_style>']

    # Save.
    write_xml_file(qgis_style, header_lines, path_out)

    return

def write_filled_contour_style_file_to_xml(path_out, contours, contour_brackets, fill_rgb_values, fill_a_values, line_rgb_values, line_a_values, outline_style, outline_width, outline_width_unit, qgis_version):
    '''
    Writes a specified set of colour-filled contours into a QGIS XML style file.
    '''

    # Store contour line properties in dictionary.
    assert outline_width_unit in ['MM', 'Pixel']
    outline_width = str(outline_width)
    line_props = {  'outline_style'     : outline_style,
                    'outline_width'     : outline_width,
                    'outline_width_unit': outline_width_unit}

    # Define header lines.
    header_lines = ['<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>']

    # Create root element.
    main_element = ET.Element('qgis') 
    main_element_dict = {
            'styleCategories'   : 'Symbology',
            'version'           : qgis_version}
    set_xml_attributes(main_element, main_element_dict)

    # Create renderer element.
    renderer = ET.SubElement(main_element, 'renderer-v2')
    renderer_dict = {
            'type'                      : 'graduatedSymbol',
            'enableorderby'             : '1',
            'attr'                      : 'ELEV',
            'graduatedMethod'           : 'GraduatedColor',

            }
    set_xml_attributes(renderer, renderer_dict)

    # Define format strings.
    color_fmt = '{:d},{:d},{:d},{:d}'

    # Write range strings for each contour level.
    ranges = ET.SubElement(renderer, 'ranges')
    n_contours = len(contours)
    for i in range(n_contours):

        # Note that when these labels are written to the XML file, the 'less
        # than' and 'greater than' symbols will be escaped.
        if i == (n_contours - 1):

            label = '{:5.0f} <= z'.format(contours[i])

        else:

            label = '{:5.0f} <= z < {:5.0f}'.format(contours[i], contours[i + 1])

        # Write the attributes of each range in the ET object.
        range_dict = {
                'lower' : '{:.12f}'.format(contour_brackets[i]),
                'symbol' : '{:d}'.format(i),
                'render' : 'true',
                'label' : label,
                'upper' : '{:.12f}'.format(contour_brackets[i + 1])}
        range_ = ET.SubElement(ranges, 'range')
        set_xml_attributes(range_, range_dict)

    # Write symbols for each contour level.
    symbols = ET.SubElement(renderer, 'symbols')
    for i in range(n_contours):

        symbol = ET.SubElement(symbols, 'symbol') 
        symbol_dict = {
                'clip_to_extent'    : '1',
                'type'              : 'fill',
                'name'              : '{:d}'.format(i),
                'alpha'             : '1' # Note: alpha value is specified elsewhere.
                }
        set_xml_attributes(symbol, symbol_dict)
        
        # Define a layer for the filled part of the symbol.
        layer = ET.SubElement(symbol, 'layer')
        layer_dict = {
                'class'     :'SimpleFill',
                }
        set_xml_attributes(layer, layer_dict)

        # Set the properties for the filled polygon.
        prop_dict = {
            'color'             : color_fmt.format(*fill_rgb_values[i, :], fill_a_values[i]),
            'outline_color'     : color_fmt.format(*line_rgb_values[i, :], line_a_values[i]),
            'style'             :'solid'
            }

        # Merge all the properties into one dictionary and write to the ET
        # object.
        prop_dict = {**prop_dict, **line_props}

        for key, value in prop_dict.items():

            prop = ET.SubElement(layer, 'prop')
            prop.set('k', key)
            prop.set('v', value)

    # Add instructions about ordering the contours in ascending order.
    orderby = ET.SubElement(renderer, 'orderby')
    orderby_clause = ET.SubElement(orderby, 'orderByClause')
    orderby_dict = {
            'nullsFirst'    : '0',
            'asc'           : '1'}
    set_xml_attributes(orderby_clause, orderby_dict)
    orderby_clause.text = '\"ELEV\"'

    # Write the XML file.
    write_xml_file(main_element, header_lines, path_out)

    return

# --- Handling RGBA. ----------------------------------------------------------
def rgb_float_to_int(in_float):
    '''
    Convert from fractional RGB values (0.0 - 1.0) to integer RGB values (0 - 255).
    '''

    out_int = np.round(in_float * 255.0)
    out_int = out_int.astype(np.int32)

    return out_int

def check_rgba_values(n, rgb_values, a_values, tol = 1.0E-12):
    '''
    Check that the input arrays have the shapes (n, 3) and (n,), and they are
    between 0 and 1.
    '''

    assert len(rgb_values.shape) == 2
    assert rgb_values.shape[0] == n
    assert rgb_values.shape[1] == 3
    assert np.all(rgb_values >= -tol)
    assert np.all(rgb_values <= (1.0 + tol))
    #
    assert len(a_values.shape) == 1 
    assert a_values.shape[0] == n 
    assert np.all(a_values >= -tol)
    assert np.all(a_values <= (1.0 + tol))

    return

def sRGBtoLin(colorChannel):
    '''
    Converts a fractional (0.0 - 1.0) sRGB gamma encoded color value to a "linearised value" (I don't know what is meant by this).
    From https://stackoverflow.com/a/56678483/6731244.
    '''

    if ( colorChannel <= 0.04045 ):

        val = colorChannel / 12.92

    else:

        val = ((colorChannel + 0.055)/1.055) ** 2.4

    return val

def Luminance_from_RGB(R, G, B):
    '''
    Converts RGB values to Luminance.
    From https://stackoverflow.com/a/56678483/6731244.
    '''

    Y = (0.2126 * sRGBtoLin(R) + 0.7152 * sRGBtoLin(G) + 0.0722 * sRGBtoLin(B))

    return Y

def YtoLstar(Y):
    '''
    Converts luminance to L* ("perceived brightness).
    Luminance value should be between 0.0 and 1.0
    Return value is between 0.0 and 100.0
    From https://stackoverflow.com/a/56678483/6731244.
    '''

    if (Y <= (216.0/24389)):

        val = Y * (24389.0/27.0)

    else:

        val = ((Y ** (1.0/3.0)) * 116) - 16.0

    return val

def get_rgba_values_two_colour_linear(contours, linear_colour_stops):
    '''
    Generates RGB values changing linearly between two end points, at specified contour points.
    '''
    
    # Unpack the start and end colours.
    colour_start, colour_end = linear_colour_stops

    # Assign linear span of values for R, G, and B.
    n_contours = len(contours)
    rgb_values = np.zeros((n_contours, 3))
    for i in range(3):

        rgb_values[:, i] = np.linspace(colour_start[i], colour_end[i],
                            num = n_contours)

    # All alpha values are 1.
    a_values = np.zeros(n_contours) + 1.0

    return rgb_values, a_values

def get_rgba_values_pyplot_linear(contours, pyplot_ramp_name):
    '''
    Generates RGB values that are sampled from a specified Matplotlib colour map at specified points.
    '''

    # Normalise the contours in the range 0 to 1.
    z_min = contours[0]
    z_max = contours[-1]
    z_range = z_max - z_min
    contours_normalised = (contours - z_min) / z_range

    # Sample the specified colour ramp at the contour points.
    c_map = matplotlib.cm.get_cmap(pyplot_ramp_name)
    rgb_values = c_map(contours_normalised)[:, :-1]

    # All alpha values are 1.
    n_contours = len(contours)
    a_values = np.zeros(n_contours) + 1.0

    return rgb_values, a_values

# --- Wrapper scripts for writing style files. --------------------------------
def make_linear_colour_ramp(n_values, rgb_min, rgb_max, dir_out, ramp_name):
    '''
    Create a colour ramp between two colour end points and save as a QGIS style file.
    '''

    # Define contour points.
    pts = np.linspace(0.0, 1.0, num = n_values)
    
    # Fill in colour values.
    a_values = np.zeros(n_values) + 1.0
    rgb_values = np.zeros((n_values, 3))
    for i in range(3):

        rgb_values[:, i] = np.linspace(rgb_min[i], rgb_max[i], num = n_values)

    # Check RGB and alpha values.
    check_rgba_values(n_values, rgb_values, a_values)

    # Define name and output path.
    path_out = os.path.join(dir_out, '{:}.xml'.format(ramp_name))

    # Write the XML file.
    write_color_ramp_to_qgis_xml(path_out, ramp_name, pts,
                                rgb_values, a_values)

    return

def make_filled_contours_style_file(z_min, d_z, n_bins, colour_ramp_type, dir_out, style_file_name, outline_contrast, outline_style, outline_width, outline_width_unit, qgis_version, linear_colour_stops = None, pyplot_ramp_name = None): 
    '''
    Create a QGIS style file that describes colour-filled contours.
    '''

    # Define name and output path.
    path_out = os.path.join(dir_out, '{:}.qml'.format(style_file_name))

    # Define contour points.
    # These should match the values of the contours generated in QGIS with
    # Raster > Contours (although it does not matter if there are extra
    # values at the ends).
    z_range = n_bins * d_z
    z_max = z_min + z_range
    n_contours = n_bins + 1
    contours = np.linspace(z_min, z_max, num = n_contours)
    
    # Define points bracketing contours.
    # For example, with 500 m contours, the contour at 1,500 m is bracketed
    # by 1,250 m and 1,750 m.
    contour_brackets = np.zeros(n_contours + 1)
    contour_brackets[: -1] = contours - (d_z / 2.0)
    contour_brackets[-1] = contours[-1] + (d_z / 2.0)

    # Define fill RGB and alpha values (ranging from 0.0 to 1.0).
    # These are the fill values for each contour.
    # For example, if the contours are [0.0, 10.0, 20.0], then the first
    # RGBA value describes the fill colour for the regions with 
    # elevation where 0.0 <= z < 10.0 and the last RGBA value describes the
    # the fill colour for the regions with elevation where z > 20.0
    if colour_ramp_type == 'two_colour_linear':

        fill_rgb_values, fill_a_values = \
                get_rgba_values_two_colour_linear(contours, linear_colour_stops)

    elif colour_ramp_type == 'pyplot_linear':

        fill_rgb_values, fill_a_values = \
                get_rgba_values_pyplot_linear(contours, pyplot_ramp_name)

    else:

        raise ValueError
    
    # Calculate the colours of the contour lines. These are chosen so that
    # there is a constant contrast between the contour lines and the contour
    # fill colour.
    #
    # First, we calculate the perceived brightness (L_star) of the contour
    # fill colours.
    fill_L_star = np.zeros(n_contours)
    for i in range(n_contours):

        fill_L_star[i] = YtoLstar(Luminance_from_RGB(*fill_rgb_values[i, :]))
    
    # Normalise perceived brightness from 0.0 to 1.0.
    fill_L_star = fill_L_star / 100.0 

    # Next, we loop through the contour lines. We find the 'local' brightness
    # (the mean of the brightness of the two fill colours either side of the
    # line). The brightness of the line is then the local brightness
    # minus the desired contrast value (except if this would lead to a
    # negative brightness, in which case it is the sum of the local brightness
    # and the desired contrast).
    #
    # The line RGB and alpha values (ranging from 0.0 to 1.0) are arrays
    # for each contour.
    # For example, if the contours are [0.0, 10.0, 20.0], then the first
    # RGBA value describes the line colour for the 0.0 m contour and the
    # last RGBA value describes the line colour for the 20.0 m contour.
    #
    line_rgb_values = np.zeros((n_contours, 3))
    for i in range(n_contours):

        # The first contour has no lower fill, so there is no need to take
        # the mean.
        if (i == 0):

            local_L_star = fill_L_star[i]

        # Get mean brightness of the two colours separated by the contour.
        else:

            local_L_star = (fill_L_star[i] + fill_L_star[i - 1]) / 2.0

        # Assign the colour of the line, by increasing or decreasing the
        # brightness relative to the surrounding colours.
        if local_L_star > outline_contrast:

            line_rgb_values[i, :] = local_L_star - outline_contrast

        else:

            line_rgb_values[i, :] = local_L_star + outline_contrast

    # Use constant 100% opacity.
    line_a_values = np.zeros(n_contours) + 1.0

    # Check RGB and alpha values.
    check_rgba_values(n_contours, fill_rgb_values, fill_a_values)
    check_rgba_values(n_contours, line_rgb_values, line_a_values)

    # Convert floating values to integer.
    fill_rgb_values = rgb_float_to_int(fill_rgb_values)
    fill_a_values   = rgb_float_to_int(fill_a_values)
    #
    line_rgb_values = rgb_float_to_int(line_rgb_values)
    line_a_values   = rgb_float_to_int(line_a_values)

    # Write to QML.
    write_filled_contour_style_file_to_xml(path_out, contours, contour_brackets,
            fill_rgb_values, fill_a_values, line_rgb_values, line_a_values,
            outline_style, outline_width, outline_width_unit, qgis_version)

    return

# --- Program control. --------------------------------------------------------
def parse_input_file(path_input_file):
    '''
    Read the input file, parse the arguments, and do some basic validation.
    '''

    # Read the input file line-by-line.
    # Each line consists of 2 or more values separated by spaces.
    # The first value is the argument name, which becomes a dictionary key
    # The remaining values are the argument values, which are stored in
    # the dictionary as a list.
    input_dict = dict()
    with open(path_input_file, 'r') as in_id:

        for line_raw in in_id:
            
            line = line_raw.split()
            if len(line) < 2:

                print('Input file: {:}'.format(path_input_file))
                print('Line: {:}'.format(line))
                print('Found {:d} arguments. Need at least 2'.format(len(line)))
                raise ValueError
            
            input_dict[line[0]] = line[1:]

    # Define the name, type and number of arguments expected for each task.
    # Each task has a dictionary of argument names. The dictionary value
    # for a given argument name is a list with two items. The first item
    # is a list of the types of that argument's values. The second item in the
    # list is either `None` or a dictionary specifying sub-arguments.
    # The sub-argument dictionary follows the same format, except it does
    # not include a dictionary of sub-sub-arguments.
    expected_args = dict()
    #
    expected_args['make_linear_colour_ramp'] = {
        'n_values'  : [[int], None],
        'rgb_min'   : [[float, float, float], None],
        'rgb_max'   : [[float, float, float], None],
        'dir_out'   : [[str], None],
        'ramp_name' : [[str], None]}
    #
    expected_args['make_filled_contours'] = {
        'z_min'             : [[float], None],
        'd_z'               : [[float], None],
        'n_bins'            : [[int],   None],
        'colour_ramp_type'  : [[str],
                               {'two_colour_linear' :
                                    {   'rgb_min' : [float, float, float],
                                        'rgb_max' : [float, float, float]},
                                'pyplot_linear' :
                                    {   'pyplot_ramp_name' : [str]}
                                        }],
        'outline_contrast'  : [[float], None],
        'outline_style'     : [[str],   None],
        'outline_width'     : [[float], None],
        'outline_width_unit': [[str],   None],
        'dir_out'           : [[str],   None],
        'qgis_version'      : [[str],   None],
        'style_file_name'   : [[str],   None]}

    # The input file must specify the task.
    assert 'task' in input_dict.keys(), 'The input file {:} did not specify the input \'task\'.'.format(path_input_file)

    # Check that the arguments provided are appropriate to the task.
    input_dict['task'] = input_dict['task'][0]
    check_args = expected_args[input_dict['task']]
    recast_args = dict()
    recast_args['task'] = input_dict['task'] 
    for key in check_args:
        
        # Check that a required argument is present.
        assert key in input_dict.keys(), 'For task {:}, the input \'{:}\' is required but was not found in the file {:}'.format(input_dict['task'], key, path_input_file)

        # Check that the number of values is correct for this argument.
        assert len(input_dict[key]) == len(check_args[key][0]), 'For task {:}, the input \'{:}\' should specify {:d} arguments, but only found {:d}.'.format(input_dict['task'], key,  len(check_args[key][0]), len(input_dict[key]))
        
        # Cast the input strings into the specified types (will raise an
        # error if not possible to cast).
        recast_arg = [f(x) for f, x in zip(check_args[key][0], input_dict[key])]

        # Convert single arguments from lists into scalars.
        if len(recast_arg) == 1:

            recast_arg = recast_arg[0]

        # Store the argument values.
        recast_args[key] = recast_arg

        # Check sub-arguments.
        if check_args[key][1] is not None:
            
            # Get the list of sub-arguments for this argument.
            check_sub_args = check_args[key][1][recast_arg]
            
            # Check that the sub-arguments provided are appropriate.
            for sub_key in check_sub_args:
                
                # Check that the required sub-argument is present.
                assert sub_key in input_dict.keys(), 'For task {:} with setting {:} = {:}, the input \'{:}\' is required but was not found in the file {:}'.format(input_dict['task'], key, recast_arg, sub_key, path_input_file)

                # Check that the number of values is correct for this sub-argument.
                assert len(input_dict[sub_key]) == len(check_sub_args[sub_key]), 'For task {:}, the input \'{:}\' should specify {:d} arguments, but only found {:d}.'.format(input_dict['task'], sub_key, len(check_sub_args[sub_key]), len(input_dict[sub_key]))

                # Cast the input strings into the specified types (will raise
                # an error if not possible to cast).
                recast_sub_arg = [f(x) for f, x in
                        zip(check_sub_args[sub_key], input_dict[sub_key])]

                # Convert single arguments from lists into scalars.
                if len(recast_sub_arg) == 1:

                    recast_sub_arg = recast_sub_arg[0]

                # Store the argument values.
                recast_args[sub_key] = recast_sub_arg

    # Check for arguments in the input file that are not required.
    superfluous_arguments = [x for x in input_dict.keys() if x not in recast_args.keys()]
    if len(superfluous_arguments) > 0:

        raise ValueError("The following arguments were found in the input file that are not required: {:}".format(superfluous_arguments)) 

    return recast_args

def main():

    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input_file", help = "File path to file with input parameters.")
    args = parser.parse_args()
    path_input_file = args.path_input_file

    # Parse the input file.
    input_args = parse_input_file(path_input_file)

    # Run the specified task.
    if input_args['task'] == 'make_linear_colour_ramp':

        make_linear_colour_ramp(
            input_args['n_values'], input_args['rgb_min'], input_args['rgb_max'],
            input_args['dir_out'], input_args['ramp_name'])

    elif input_args['task'] == 'make_filled_contours':
        
        if input_args['colour_ramp_type'] == 'two_colour_linear':

            linear_colour_stops = [input_args['rgb_min'], input_args['rgb_max']]
            pyplot_ramp_name = None


        elif input_args['colour_ramp_type'] == 'pyplot_linear':

            linear_colour_stops = None
            pyplot_ramp_name = input_args['pyplot_ramp_name']

        make_filled_contours_style_file(
            input_args['z_min'], input_args['d_z'], input_args['n_bins'],
            input_args['colour_ramp_type'], input_args['dir_out'],
            input_args['style_file_name'], input_args['outline_contrast'],
            input_args['outline_style'], input_args['outline_width'],
            input_args['outline_width_unit'],
            input_args['qgis_version'],
            linear_colour_stops = linear_colour_stops,
            pyplot_ramp_name = pyplot_ramp_name)

    else:

        raise ValueError

    # Put QGIS version into input argument.

    return

if __name__ == '__main__':

    main()
