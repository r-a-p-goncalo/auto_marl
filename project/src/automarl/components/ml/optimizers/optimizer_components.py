from automarl.component import Component, ParameterSignature, requires_input_process
from automarl.core.advanced_input_management import ComponentParameterSignature, LookableParameterSignature
from automarl.components.ml.models.model_components import ModelComponent
from automarl.components.basic_components.dynamic_value import get_value_or_dynamic_value
from automarl.core.advanced_input_utils import get_value_of_type_or_component
from automarl.components.loggers.logger_component import ComponentWithLogging
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from abc import abstractmethod

class OptimizerSchema(Component):
    
    parameters_signature = {"model" : ComponentParameterSignature(mandatory=False, ignore_at_serialization=True),
                            "params" : ParameterSignature(mandatory=False, ignore_at_serialization=True)}
    
    
    exposed_values = {
        "optimizations_done" : 0
    }
    
    
    def _process_input_internal(self):
        super()._process_input_internal()
        
        self.model : ModelComponent = self.get_input_value("model")
        self.params = self.get_input_value("params")

        if self.model is not None and self.params is not None:
            raise Exception(f"Can only use either params or model in optimizer")
        
        elif self.model is None and self.params is None:
            raise Exception(f"Either model or params should be used in optimizer")
        
        elif self.model is not None:
            self.params = self.model.get_model_params() #gets the model parameters to optimize


    def set_params(self, new_params):
        self.params = new_params
        self.input["params"] = new_params
        self.input["model"] = None

        
        
    @abstractmethod
    def optimize_model(self, predicted, correct) -> None:
        
        '''
            Optimizes the model based on the prediction(s) and correct value(s)
            
            Args:
                predicted is the prediced value(s) of the model for a given input
                correct is the correct value(s) for the given input
                
        '''
        
        pass

    def optimize_with_loss(self, loss):
        pass

    def clear_optimizer_gradients(self):
        pass

    def optimize_with_backward_pass_done(self):

        self._optimize_with_backward_pass_done()

        self.values["optimizations_done"] = self.values["optimizations_done"] + 1

    def _optimize_with_backward_pass_done(self):
        pass


class AdamOptimizer(OptimizerSchema, ComponentWithLogging):

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                       "learning_rate" : ParameterSignature(
                           default_value=0.001,
                           custom_dict={"hyperparameter_suggestion" : [ "float", {"low": 1.5e-8, "high": 9e-2 }]}
                           ),

                       "amsgrad" : ParameterSignature(default_value=False),

                       "clip_grad_value" : ParameterSignature(
                            mandatory=False, 
                            description="If defined, it clips the gradients to the given value",
                            custom_dict={
                                "hyperparameter_suggestion" : [ "float", {"low": 0.05, "high": 0.5 }]}
                        ),

                       "clip_grad_norm" : ParameterSignature(
                           mandatory=False, 
                           description="If defined, it clips the gradients to the given norm"
                        ),

                       "linear_decay_learning_rate_with_final_input_value_of" : LookableParameterSignature(
                           mandatory=False
                        )
                       
                       
                    }    
    
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()

        self.lg.writeLine(f"Starting to process optimizer input")

        self.lr = self.get_input_value("learning_rate")
        self.amsgrad = self.get_input_value("amsgrad")
                
        self.torch_adam_opt = optim.Adam(params=self.params,lr=self.lr, amsgrad=self.amsgrad)

        self._initialize_decays()
        self._initialize_grad_clip()

        self.lg.writeLine(f"Finished processing optimizer input")

    # INITIALIZATION --------------------------------------------------------------------------

    def _initialize_decays(self):

        self.lr_scheduler = None

        self.linear_decay_learning_rate_with_final_input_value_of = self.get_input_value("linear_decay_learning_rate_with_final_input_value_of", accepted_types=(int)) 

        if self.linear_decay_learning_rate_with_final_input_value_of != None:

            for group in self.torch_adam_opt.param_groups:
                if "initial_lr" not in group:
                    group["initial_lr"] = group["lr"] # we set the initial in case we're resuming training

            self.lg.writeLine(f"LR will have linear decay, using a linear decay till 0 and a predicted final number of optimizations of {self.linear_decay_learning_rate_with_final_input_value_of}")

            self.lr_scheduler = LambdaLR(self.torch_adam_opt, last_epoch=self.values["optimizations_done"], lr_lambda=lambda step: max(0.0, 1 - step / self.linear_decay_learning_rate_with_final_input_value_of))


    def _initialize_grad_clip(self):

        self.clip_grad_value = get_value_of_type_or_component(self, "clip_grad_value", float)

        if self.clip_grad_value != None:
            self.lg.writeLine(f"Optimizer has clip grad value of type: {type(self.clip_grad_value)}, with value {self.clip_grad_value}")
        
        self.clip_grad_norm = get_value_of_type_or_component(self, "clip_grad_norm", float)

        if self.clip_grad_norm != None:
            self.lg.writeLine(f"Optimizer has clip grad norm of type: {type(self.clip_grad_norm)}, with value {self.clip_grad_norm}")

    # EXPOSED METHODS --------------------------------------------------------------------------
    
    @requires_input_process
    def optimize_model(self, predicted, correct) -> None:
                    
        # Compute Huber loss TODO : The loss should not be hard calculated like this
        criterion = nn.SmoothL1Loss()
        loss = criterion(predicted, correct)
        
        self.optimize_with_loss(loss)


    @requires_input_process
    def optimize_with_loss(self, loss):

        self.clear_optimizer_gradients()

        #Optimize the model
        loss.backward()

        self.optimize_with_backward_pass_done()


    @requires_input_process
    def clear_optimizer_gradients(self):
        self.torch_adam_opt.zero_grad() #clears previous accumulated gradients in the optimizer


    def _apply_clipping(self):

        '''Applies in place gradient clipping if any'''

        if self.clip_grad_value != None:
            nn.utils.clip_grad_value_(self.params, get_value_or_dynamic_value(self.clip_grad_value))
        
        if self.clip_grad_norm != None:
            nn.utils.clip_grad_norm_(self.params, get_value_or_dynamic_value(self.clip_grad_norm))
    
    
    def _optimize_with_backward_pass_done(self):

        self._apply_clipping()
        
        self.torch_adam_opt.step()
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        
        

class SimpleSGDOptimizer(OptimizerSchema):

    parameters_signature = {
        "learning_rate": ParameterSignature(default_value=0.01)
    }

    def _process_input_internal(self):
        super()._process_input_internal()
        
        self.lr = self.get_input_value("learning_rate")

        # Define SGD optimizer
        self.sgd_optimizer = optim.SGD(self.params, lr=self.lr)

    @requires_input_process
    def optimize_model(self, predicted, correct) -> None:
        super().optimize_model(predicted, correct)

        # Use simple Mean Squared Error loss
        criterion = nn.MSELoss()
        loss = criterion(predicted, correct)

        # Zero previous gradients
        self.sgd_optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model parameters
        self.sgd_optimizer.step()