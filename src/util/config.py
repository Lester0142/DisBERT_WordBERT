import os
from collections import OrderedDict

from configparser import ConfigParser

class Config:
    """
    A class for managing configuration files using ConfigParser.
    Supports reading, updating, retrieving, and saving configuration parameters.
    """
    def __init__(self, main_conf_path):
        """
        Initializes the Config class.

        :param main_conf_path: Path to the main configuration directory.
        """
        self.main_conf_path = main_conf_path        
        self.model_config = self.read_config(os.path.join(main_conf_path, 'model_config.cfg'))

    def read_config(self, conf_path):
        """
        Reads the configuration file and stores its contents as an ordered dictionary.

        :param conf_path: Path to the configuration file.
        :return: Ordered dictionary of configuration sections and parameters.
        """
        conf_dict = OrderedDict() 
        config = ConfigParser()
        config.read(conf_path)
        
        for section in config.sections():
            section_config = OrderedDict(config[section].items())
            conf_dict[section] = self.type_ensurance(section_config)
            self.__dict__.update(conf_dict[section])

        return conf_dict

    def ensure_value_type(self, v):
        """
        Ensures the correct type for a configuration value.

        :param v: The value to be type-checked.
        :return: The value converted to the appropriate type.
        """
        BOOLEAN = {'false': False, 'False': False,
                   'true': True, 'True': True}
        if isinstance(v, str):
            try:
                value = eval(v)
                if not isinstance(value, (str, int, float, list, tuple)):
                    value = v
            except:
                value = BOOLEAN.get(v, v)
        else:
            value = v
        return value

    def type_ensurance(self, config):
        """
        Ensures all configuration values are of the correct type.

        :param config: Dictionary of configuration parameters.
        :return: Dictionary with type-converted values.
        """
        return {k: self.ensure_value_type(v) for k, v in config.items()}

    def get_param(self, section, param):
        """
        Retrieves a parameter from the configuration.

        :param section: The section name in the config file.
        :param param: The parameter name within the section.
        :return: The parameter value.
        :raises NameError: If section or parameter is not found.
        """
        if section not in self.model_config:
            raise NameError(f"There is no section named '{section}'")
        if param not in self.model_config[section]:
            raise NameError(f"There is no parameter named '{param}'")
        
        return self.model_config[section][param]

    def update_params(self, params):
        """
        Updates configuration parameters dynamically.

        :param params: Dictionary of parameters to update.
        """
        for k, v in params.items():
            updated = False
            for section in self.model_config:
                if k in self.model_config[section]:
                    self.model_config[section][k] = self.ensure_value_type(v)
                    self.__dict__[k] = self.model_config[section][k]
                    updated = True
                    break
            if not updated:
                print(f"Parameter not updated. '{k}' does not exist.")

    def save(self, base_dir):
        """
        Saves the current configuration to a file.

        :param base_dir: Directory where the configuration file should be saved.
        """
        def helper(section_k, section_v):
            sec_str = f"[{section_k}]\n"
            sec_str += "\n".join(f"{k}={v}" for k, v in section_v.items()) + "\n\n"
            return sec_str
        
        main_conf_str = "".join(helper(section, self.model_config[section]) for section in self.model_config)
        with open(os.path.join(base_dir, 'model_config.cfg'), 'wt') as f:
            f.write(main_conf_str)
        print(f"Main config saved in {base_dir}")

    def __getitem__(self, item):
        """
        Retrieves a section from the configuration using indexing.

        :param item: Section name.
        :return: Ordered dictionary of the section parameters.
        :raises NameError: If section does not exist.
        """
        if not isinstance(item, str):
            raise TypeError("Index must be a string")
        if item not in self.model_config:
            raise NameError(f"There is no section named '{item}'")
        return self.model_config[item]

    def __str__(self):
        """
        Returns a formatted string representation of the configuration.

        :return: String representation of the configuration.
        """
        config_str = '\n========== Model Config ==========\n'
        for section, params in self.model_config.items():
            config_str += f"[{section}]\n" + "\n".join(f"{k}: {v}" for k, v in params.items()) + "\n\n"
        return config_str
