"""
Helper methods for C5.0 parsing
"""

import json
import re

int_and_float_re = re.compile("^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")
bool_re = re.compile("((True)|(False))")


def str_to_bool(bool_str):
    if bool_str == 'True':
        return True
    elif bool_str == 'False':
        return False


def parse_variable_str_to_dict(variables_str):
    """
    Parse string of variables of the form
    'variable_name="val" variable_name="val" variable_name="val"'
    into dict

    Where variable vals are cast to the correct type.
    This is the form C5 stores output data
    """
    variables = {}

    for var_str in variables_str.split(' '):
        if var_str != '':

            var_name = var_str.split('=')[0]
            var_value = var_str.split('=')[1].replace('"', '')

            # Cast to correct type
            if re.match(int_and_float_re, var_value):
                var_value = json.loads(var_value)
            elif re.match(bool_re, var_value):
                var_value = str_to_bool(var_value)

            variables[var_name] = var_value

    return variables
