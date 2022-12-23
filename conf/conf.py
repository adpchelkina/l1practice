import logging
from dynaconf import Dynaconf

logging.basicConfig(level=logging.INFO)

settings = Dynaconf(settings_file="conf/settings.toml")
