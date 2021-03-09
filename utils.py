import inspect

"""
This function generates default parameters of a module, but does not generate positional arguments
"""


def generate_default_configuration(factory_function, item_name):
    default_factory = factory_function(item_name)
    default_values = {
        param.name: param.default
        for param in inspect.signature(default_factory).parameters.values()
        if param.default != inspect.Parameter.empty
    }
    return default_values