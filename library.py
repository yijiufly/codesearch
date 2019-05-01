from binary import Binary
class Library(Binary):
    def __init__(self, path):
        self.generatefuncNameFull(path)
        #self.generatefuncNameFilted(path2)
        self.getGraphFromPath(path)
        self.funcName2MatchingLabel = dict()
        self.libraryName = None
