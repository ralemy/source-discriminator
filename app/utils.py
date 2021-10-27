from datetime import datetime
import sys

def loggable(cls):
    def log(self, *args):
        if hasattr(self, 'silent') and self.silent: 
            return 
        print(datetime.now(), self.__class__.__name__, *args)
    def error(self,*args):
        print(datetime.now(), self.__class__.__name__, *args, file=sys.stderr)
    def debug(self, *args):
        print('=' * 30)
        print(*args)
    setattr(cls, 'log', log)
    setattr(cls,'error', error)
    setattr(cls, 'debug', debug)
    return cls

INDEV = True

def generate_session_id():
    return datetime.now().strftime('%Y-%m-%d_%H_%M_%s')


