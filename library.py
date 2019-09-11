from binary import Binary
class Library(Binary):
    def __init__(self, libraryName, dotPath, embFile, namFile, filter_size=0):
        self.libraryName = libraryName
        self.loadCallGraph(dotPath)
        self.generatefuncNameFilted(namFile, filter_size)
        self.getGraphFromPathfilted()
        self.embFile = embFile
