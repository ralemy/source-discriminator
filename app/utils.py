from datetime import datetime

def loggable(cls):
    def log(self, *args):
        if hasattr(self, 'silent') and self.silent: 
            return 
        print(datetime.now(), *args)
    setattr(cls, 'log', log)
    return cls

INDEV = True

def generate_session_id():
    return datetime.now().strftime('%Y-%m-%d_%H_%M_%s')


