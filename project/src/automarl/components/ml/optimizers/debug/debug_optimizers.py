from automarl.components.loggers.debug.component_with_logging_debug import ComponentDebug
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer
from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.component import requires_input_process


class AdamOptimizerDebug(AdamOptimizer, ComponentDebug):

    is_debug_schema = True

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                       }    
    
    exposed_values = {
    }
    
    
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()

        self.lg.open_or_create_relative_folder("clips")

    # EXPOSED METHODS --------------------------------------------------------------------------
    
    def optimize_with_loss(self, loss):

        self.lg.writeLine(f"Optimization {self.values['optimizations_done']}: Optimizing with loss: {loss}", file="loss_optimization.txt", use_time_stamp=False)

        super().optimize_with_loss(loss)

    def _apply_clipping(self):

        '''Applies in place gradient clipping if any'''


        if self.clip_grad_value != None or self.clip_grad_norm != None:

            path_to_write = self.lg.new_relative_path_if_exists("clip.txt", dir="clips")

            self.lg.writeLine(f"\nGRADIENTS BEFORE CLIPPING:\n", file=path_to_write, use_time_stamp=False)

            for name, p in self.model.model.named_parameters():

                self.lg.writeLine(f"Parameter named {name}: ", file=path_to_write, use_time_stamp=False)

                if p.grad is None:
                    self.lg.writeLine(f"Has no gradients", file=path_to_write, use_time_stamp=False)
                else:
                    gradients = p.grad.detach().cpu().numpy()
                    self.lg.writeLine(f"Has gradients with len: {len(gradients)}", file=path_to_write, use_time_stamp=False)
                    self.lg.writeLine(f"{gradients}", 
                                      file=path_to_write, use_time_stamp=False)

                self.lg.writeLine(f"", file=path_to_write, use_time_stamp=False)

        super()._apply_clipping()

        if self.clip_grad_value != None or self.clip_grad_norm != None:
            self.lg.writeLine(f"\nGRADIENTS AFTER CLIPPING:\n", file=path_to_write, use_time_stamp=False)

            for name, p in self.model.model.named_parameters():

                self.lg.writeLine(f"Parameter named {name}: ", file=path_to_write, use_time_stamp=False)

                if p.grad is None:
                    self.lg.writeLine(f"Has no gradients", file=path_to_write, use_time_stamp=False)
                else:
                    gradients = p.grad.detach().cpu().numpy()
                    self.lg.writeLine(f"Has gradients with len: {len(gradients)}", file=path_to_write, use_time_stamp=False)
                    self.lg.writeLine(f"{gradients}", 
                                      file=path_to_write, use_time_stamp=False)
        
    