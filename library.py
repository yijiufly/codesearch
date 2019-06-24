from binary import Binary
class Library(Binary):
    def __init__(self, libraryName, dotPath, embFile):
        self.libraryName = libraryName
        self.loadCallGraph(dotPath)
        self.getGraphFromPath()
        self.embFile = embFile
