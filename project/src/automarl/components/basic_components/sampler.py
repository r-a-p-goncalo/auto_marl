

from automarl.component import Component


class Sampler(Component):
    
    '''
    Samples a Component
    '''
    
    parameters_signature = {
                    }    

    def sample(self) -> Component:
        pass