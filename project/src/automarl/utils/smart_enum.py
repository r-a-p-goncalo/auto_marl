

from enum import Enum, EnumMeta

from automarl.serialization.custom_json_logic import CustomJsonLogic, register_custom_strategy

class SmartEnumMeta(EnumMeta):
    def __repr__(cls):
        return f"<class '{cls.__module__}.{cls.__qualname__}'>"

class SmartEnum(Enum, metaclass=SmartEnumMeta):

    @classmethod
    def clone(cls, value):
        return cls.from_value(cls.from_value(value).value)

    @classmethod
    def from_value(cls, value):

        if isinstance(value, cls):
            return value
        
        if isinstance(value, int):
            return cls(value)
        
        if isinstance(value, str):

            name = value.strip().upper()
            if name in cls.__members__:
                return cls[name]
            
            if name.lstrip('-').isdigit():
                return cls(int(name))
            
        raise ValueError(f"Cannot convert {value!r} to {cls.__name__}")
    
    @classmethod
    def equals_value(cls, value_fst, value_scd):
        value_fst = cls.from_value(value_fst)
        value_scd = cls.from_value(value_scd)

        return value_fst == value_scd
    

class SmartEnumEncoderDecoder(CustomJsonLogic):

    def to_dict(value : SmartEnum):
        return {"value" : value.name}
    
    def from_dict(dict, element_type : type[SmartEnum], decode_elements_fun, source_component):
        
        return element_type.from_value(dict["value"])
    
register_custom_strategy(SmartEnum, SmartEnumEncoderDecoder)
    