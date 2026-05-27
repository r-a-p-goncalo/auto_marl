

from automarl.component import Component


class Event:
    
    
    def __init__(self):
        self.subscriptions = []

    def subscribe(self, fn):
        self.subscriptions.append( fn)
        
        

    def notify(self, *args, **kwargs):
        for fn in self.subscriptions:
            fn(*args, **kwargs)



    
class EventfulComponent(Component):
    
    '''A component with events'''
    
    STATIC_EVENTS : dict[str, Event] = {} # static events of component
    
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.EVENTS = {**self.STATIC_EVENTS} # initialized events attribute of component, with access to the 
    
    def subscribe_event(self, event_key, fn):
        self.EVENTS[event_key].subscribe(fn)


    def activate_event(self, event_key):
        self._event_activated(event_key)
        
        
    def _event_activated(self, event_key):
        self.EVENTS[event_key].notify()