from binary import Binary
class Library(Binary):
    def __init__(self, libraryName, dotPath, embFile):
        self.libraryName = libraryName
        self.generatefuncNameFull(dotPath)
        self.getGraphFromPath(dotPath)
        self.embFile = embFile
