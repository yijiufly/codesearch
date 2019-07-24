from binary import Binary
class Library(Binary):
    def __init__(self, libraryName, dotPath, embFile, namFile):
        self.libraryName = libraryName
        self.loadCallGraph(dotPath)
        self.generatefuncNameFilted(namFile)
        self.getGraphFromPathfilted()
        self.embFile = embFile
