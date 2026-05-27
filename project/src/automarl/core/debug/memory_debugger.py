import os
from automarl.components.loggers.debug.component_with_logging_debug import ComponentDebug
from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.component import Component
import psutil

def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss, mem_info.vms


class MemoryDebuggerComponent(ComponentDebug):
    
    is_debug_schema = True
    
    parameters_signature = {
                    }    
    
    exposed_values = {
        
        "last_evaluation" : {}
    
    }

    def _process_input_internal(self):

        prev_rss, prev_vms = process_memory()
        
        super()._process_input_internal()

        pos_rss, pos_vms = process_memory()

        self.lg.writeLine(f"Prev: RSS: {prev_rss}, VSS: {prev_vms}", file="memory_debug.txt")
        self.lg.writeLine(f"POS: RSS: {pos_rss}, VSS: {pos_vms}", file="memory_debug.txt")
        self.lg.writeLine(f"USED: RSS: {pos_rss - prev_rss}, VSS: {pos_vms - prev_vms}\n", file="memory_debug.txt")

        

