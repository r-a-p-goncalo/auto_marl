from automarl.component import Component, ParameterSignature, requires_input_process

from automarl.core.advanced_input_management import ComponentParameterSignature
from automarl.components.ml.models.model_components import ModelComponent

from automarl.utils.class_util import get_class_from

from automarl.utils.shapes_util import reduce_space_dimension
from automarl.components.loggers.logger_component import ComponentWithLogging

import torch

class PolicyInterface(Component):
    
    '''
    A policy interface, defining only the methods necessary for a policy
    This abstracts wrappers and our own implemented strategies
    '''
        
    def get_policy_output_shape(self):
        raise NotImplementedError()
    
    def get_policy_input_shape(self):
        raise NotImplementedError()
        
    def predict(self, state):
        raise NotImplementedError()
    
    def random_prediction(self, state):
        raise NotImplementedError()



class Policy(PolicyInterface, ComponentWithLogging):
        
    '''
    It abstracts the usage of a model for the agent in determining its actions

    A  policy has 3 processing moments:
        Computing the model output
        Computing a value which is directly tied to the action (essentially the action value without being normalized)
        Computing the action value passed to the environment
    '''
        
    parameters_signature = {
        
        "model" : ComponentParameterSignature(mandatory=False),
        
        "state_shape": ParameterSignature(ignore_at_serialization=True),
        "action_shape": ParameterSignature(ignore_at_serialization=True),
        "device" : ParameterSignature(get_from_parent=True, ignore_at_serialization=True)
    }   

    exposed_values = {"model" : 0}

    def _process_input_internal(self):
        
        super()._process_input_internal()

        self.lg.writeLine(f"Processing policy input...\n")
        
        self.input_state_shape = self.get_input_value("state_shape")
        self.output_action_shape = self.get_input_value("action_shape")

        self.lg.writeLine(f"Policy input shape is: {self.input_state_shape}")
        self.lg.writeLine(f"Policy (action) output shape is {self.output_action_shape}")

        self.device = self.get_input_value("device")

        self._initialize_model()

        self._setup_model()

        self.lg.writeLine(f"Finished processing policy input\n")

    def _compute_model_output_shape(self):
        self.model_output_shape = self.output_action_shape
        
        
    def _setup_model(self):

        self._compute_model_output_shape()

        self.model.pass_input({
            "input_shape" : self.input_state_shape["observation"], 
            "output_shape" : self.model_output_shape, 
            "device" : self.device
            }) 
        
    def _initialize_model(self):

        self.lg.writeLine(f"Initializing policy model...")
        
        if hasattr(self, "model"):

            self.lg.writeLine(f"Model already in attributes, using that one...")

        elif self.values["model"] != 0:

            self.lg.writeLine(f"Model for policy already in values, using that one...")
            self.model : ModelComponent = self.values["model"]

        else:

            self.lg.writeLine(f"No model in attributes nor defined in values, using model in input...")

            self.model = self.get_input_value("model")

            if self.model == None:
                self.lg.writeLine(f"No model found in attributes")
                raise Exception(f"Policy had no model in input nor saved in its values")
            

        self.lg.writeLine(f"Ended policy model setup")
        self.values["model"] = self.model


    @requires_input_process
    def get_model(self):
        return self.model
    
    @requires_input_process
    def get_policy_output_shape(self):
        return self.output_action_shape
    

    @requires_input_process
    def predict_model_output(self, state):
        '''
        Uses the model to process the state and compute its output
        '''

        return self.model.predict(state["observation"])
    
    
    @requires_input_process
    def get_action_val_shape(self):
        return self.output_action_shape


    @requires_input_process 
    def get_action_from_action_val(self, action_val):
        return action_val
    

    @requires_input_process
    def get_action_val_from_model_output(self, model_output, state):
        return model_output


    @requires_input_process
    def predict(self, state):
        
        '''
        Uses the state and the policy's model to predict an action
        Returns the action value for each of the passed states in a tensor
        '''
        
        model_output = self.predict_model_output(state)     
        
        action_val = self.get_action_val_from_model_output(model_output, state)

        return self.get_action_from_action_val(action_val)





        
    
    @requires_input_process
    def random_prediction(self, state):    

        return self.output_action_shape.sample()
    