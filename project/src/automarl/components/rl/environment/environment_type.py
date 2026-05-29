

from automarl.utils.smart_enum import SmartEnum


class EnvironmentType(SmartEnum):
    SINGLE = 1
    AEC = 2
    PARALLEL = 3