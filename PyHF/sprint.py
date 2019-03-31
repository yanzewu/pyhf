
SILENT = 0
MINIMAL = 1
NORMAL = 2
FULL = 3

class SPrinter:

    OP_STDOUT = 0
    OP_STR = 1

    global_level = NORMAL

    def __init__(self, output=OP_STDOUT):
        self.output = output
        self.cache = ''

    def printf(self, level, *args):
        if level <= SPrinter.global_level:
            if self.output == SPrinter.OP_STDOUT:
                print(*args)
            else:
                self.cache += ''.join(args) + '\n'

    def warning(self, *args):
        self.printf(MINIMAL, *args)

    def info(self, *args):
        self.printf(NORMAL, *args)

    def debug(self, *args):
        self.printf(FULL, *args)
        
    def text(self):
        return self.cache
