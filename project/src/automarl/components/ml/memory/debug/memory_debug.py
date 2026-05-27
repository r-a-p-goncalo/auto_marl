from automarl.components.loggers.debug.component_with_logging_debug import ComponentDebug
from automarl.components.loggers.logger_component import DEBUG_LEVEL, ComponentWithLogging
from automarl.components.ml.memory.memory_components import MemoryComponent
from automarl.component import requires_input_process


class MemoryDebug(MemoryComponent, ComponentDebug):

    is_debug_schema = True


    def _process_input_internal(self):

        super()._process_input_internal()

        self.lg.writeLine(f"Field names are: {self.field_names}")
    
    
    def push(self, transition):   

        str_pushing = ' '

        for field_name in self.field_names:

            str_value = f"\n    {field_name}: {transition[field_name]}"

            if len(str_value) > 103:
                str_value = str_value[:50] + '...' + str_value[len(str_value) - 50:]

            elif len(str_value) < 50:
                str_value = str_value + (' ' * (103 - len(str_value))) 

            if hasattr(transition[field_name], "shape"):
                str_value += f"(shape {transition[field_name].shape})"
                        
            str_pushing = f"{str_pushing}{str_value}"

        self.lg.writeLine(f"Pushing: {str_pushing}", file="pushed_transitions.txt", use_time_stamp=False, level=DEBUG_LEVEL.DEBUG) 

        super().push(transition)




    @requires_input_process
    def sample(self, batch_size):

        batch = super().sample(batch_size)

        self.lg.writeLine(f"\nSampling:\n", file="on_sample.txt", use_time_stamp=False)
        
        for i in range(batch_size):

            str_sampling = ' '

            for field_name in self.field_names:
            
                str_value = batch[field_name][i]

                if len(str_value) > 15:
                    str_value = str_value[:7] + '...' + str_value[len(str_value) - 7:]

                str_sampling = f"{str_sampling}{field_name}: {str_value} "

            self.lg.writeLine(f"{i}: {str_sampling}", file="on_sample.txt", use_time_stamp=False, level=DEBUG_LEVEL.DEBUG)


        return batch

    @requires_input_process
    def clear(self):
        
        self.lg.writeLine(f"\nBefore cleaning ({len(self)}) transitions, noting their values...\n", file="on_clear_transitions.txt", use_time_stamp=False)

        #for i in range(len(self)):
#
        #    str_pushing = ' '
    #
        #    for field_name in self.field_names:
        #                    
        #        str_pushing = f"{str_pushing}{field_name}: {self.transitions[field_name][i][:30]} " # TODO: there should be a clear way to get a single transition
    #
        #    self.lg.writeLine(f"{str_pushing}", file="on_clear_transitions.txt", use_time_stamp=False, level=DEBUG_LEVEL.DEBUG)

        super().clear()

        self.lg.writeLine(f"\After cleaning ({len(self)}) transitions\n", file="on_clear_transitions.txt", use_time_stamp=False)

