import os
import pickle as p

def loadFiles(PATH, ext=None):  # use .ida or .emb for ida file and embedding file
    filenames = []
    filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    return filenames

if __name__ == '__main__':
    path = '/home/yijiufly/Downloads/codesearch/data/openssl'
    for folder in os.listdir(path):
        namfiles = loadFiles(os.path.join(path, folder), '.nam')
        for namefile in namfiles:
            print(folder + ' ' + namefile)
            file = p.load(open(os.path.join(path, folder, namefile), 'rb'))
            p.dump(file, open(os.path.join(path, folder, namefile), 'wb'), protocol=2)
