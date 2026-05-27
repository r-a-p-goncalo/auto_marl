import copy
from automarlcore.input_management import ParameterSignature, fuse_input_signatures
from abc import ABCMeta


    
#TODO: this should make all pre computation necessary for input processing
class Schema(ABCMeta): # the meta class of all component classes, defines their behavior (not the behavior of the instances)
    
    def __init__(self_class, name, bases, namespace):
        
        # Create the schema, with its defined attributes
        super().__init__(name, bases, namespace)

        self_class.__save_original_parameters_signature()
        self_class.__reorganize_mro_so_debug_classes_come_last(bases)
        self_class.__set_exposed_values_with_super(bases)
        self_class.__fuse_parameter_signatures_with_super(bases)
        self_class.__setup_default_values_in_parameter_signatures()
        self_class.__setup_organized_parameters_signature()

        if not hasattr(self_class, "default_name") or self_class.default_name is None:
            self_class.default_name = self_class.__name__

    def __prepare__(cls_name, bases, **kwds): #all Schemes have a parameter_signature and exposed_values
        
        return {
            "parameters_signature": {},
            "original_parameters_signature" : {},
            "fused_parameters_signature" : {},
            "exposed_values": {},
            "is_debug_schema" : False,
            "target_debug_class" : None,
            "default_name" : None
        }
    
    def __save_original_parameters_signature(self_class):
        self_class.original_parameters_signature = {**self_class.parameters_signature}
    

    def __reorganize_mro_so_debug_classes_come_last(self_class, bases):

        # TODO: REORGANIZE BASES INSTEAD OF RAISING EXCEPTION

        found_non_debug_base = False
        for base_class in bases:

            if hasattr(base_class, "is_debug_schema") and base_class.is_debug_schema:
                self_class.is_debug_schema = True

                #if found_non_debug_base:
                #    raise Exception(f"When generating Schema {self_class}: found a non debug schema as parent class before debug schema {base_class}")

            else:
                found_non_debug_base = True
        

    def __set_exposed_values_with_super(self_class, bases):
        
        '''
        Updates the exposed values with super classes for this squema
        This assumes those squemas also had their exposed values processed, and so it only needs to deal with the super
        '''

        for base_class in bases:

            self_class.exposed_values =  {**self_class.exposed_values, **base_class.exposed_values}



    def __fuse_parameter_signatures_with_super(self_class, bases):
        
        '''
        Updates the input signatures with super classes for this squema
        '''

        self_class.fused_parameters_signature : dict[str, ParameterSignature]  = {**self_class.original_parameters_signature}

        self_fused_parameters_signature : dict[str, ParameterSignature] = self_class.fused_parameters_signature

        # for each of the explicitly defined super classes
        for base_class in bases:

            # we look into its parameters signature
            base_class_fused_parameters_signature : dict[str, ParameterSignature] = base_class.fused_parameters_signature

            for input_key in base_class_fused_parameters_signature.keys():

                base_class_signature = base_class_fused_parameters_signature[input_key]

                # if input key 
                if input_key in self_fused_parameters_signature.keys():

                    self_signature = self_fused_parameters_signature[input_key]

                    try:
                        self_fused_parameters_signature[input_key] = fuse_input_signatures(base_class_signature, self_signature)

                    except Exception as e:

                        raise Exception(f"Expcetion when fusing input signature with key '{input_key}' from base schema {base_class} into schema {self_class}: {e}") from e

                else:

                    self_fused_parameters_signature[input_key] = base_class_signature.clone()



    def __setup_default_values_in_parameter_signatures(self_class):

        self_parameters_signature : dict[str, ParameterSignature] = self_class.parameters_signature
        self_fused_parameters_signature : dict[str, ParameterSignature] = self_class.fused_parameters_signature

        for parameter_signature_key in self_fused_parameters_signature.keys():

            self_parameters_signature[parameter_signature_key] = self_fused_parameters_signature[parameter_signature_key].clone()
            self_parameters_signature[parameter_signature_key].setup_default_values()

    
    def __setup_organized_parameters_signature(self_class):

        '''
        Sets up the parameters signatures organized by priority
        '''

        self_class.parameters_signature_priorities : list[int] = [] #all priorities defined
        self_class.organized_parameters_signatures : dict[int, list[ParameterSignature]] = {} #ParameterSignatures organized by priorities

        self_parameters_signature : dict[str, ParameterSignature] = self_class.parameters_signature

    
        for key, parameter_signature in self_parameters_signature.items():
            
            priority = parameter_signature.priority
            
            if not priority in self_class.parameters_signature_priorities: #if this is the first key for that priority, initialize the list for that priority
                self_class.parameters_signature_priorities.append(priority)
                self_class.organized_parameters_signatures[priority] = {}
                
            self_class.organized_parameters_signatures[priority][key] = parameter_signature #put its key, parameter_signature pair in the list of respective priority  

        self_class.parameters_signature_priorities.sort()