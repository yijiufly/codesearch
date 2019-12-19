import pydot
import pickle as p
def addattributetograph():
    path = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test2/idafiles/0acc5283147612b2abd11d606d5585ac8370fc33567f7f77c0b312c207af3bf9/nginx-{openssl-0.9.8r}{zlib-1.2.9}.dot'
    graph = pydot.graph_from_dot_file(path)
    edgeList = graph.get_edge_list()
    f=open('/home/yijiufly/codesearch/edge_1.1.0b', 'r')
    for line in f.readlines():
        src=line.split()[0]
        des=line.split()[1]
        for edges in edgeList:
            if edges.get_source() == src and edges.get_destination() == des:
                #print edges.obj_dict['attributes']
                edges.obj_dict['attributes']['weight']=2

    #print graph.to_string()

    path2 = '/home/yijiufly/Downloads/codesearch/data/openssl/openssl-OpenSSL_0_9_8y/libcrypto.so.ida.nam'
    #path2 = '/home/yijiufly/Downloads/codesearch/data/zlib/idafiles/b5a15be74ed64869c5a9bc0ae9109283794bf95608674b8504a247a815e01823/libz.so.1.2.9.ida.nam'
    namelist = p.load(open(path2,'r'))
    #graph = pydot.graph_from_dot_file(path)
    nodeList = graph.get_node_list()

    for line in nodeList:
        label = line.obj_dict['attributes'].get('label', None)
        if label is not None:
            label = label.strip('\"').split('\\l')[0]
            if label in namelist:
                line.obj_dict['attributes']['class']=2
                #print label
            else:
                line.obj_dict['attributes']['class']=1
        else:
            line.obj_dict['attributes']['class']=1
    print graph.to_string()


def addattributeforBP():
    path = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx/nginx-{openssl-0.9.8x}{zlib-1.2.7.2}/nginx-{openssl-0.9.8x}{zlib-1.2.7.2}_bn.dot'
    graph = pydot.graph_from_dot_file(path)
    nodeList = graph.get_node_list()
    label1 = 'libcryptoopenssl-OpenSSL_0_9_8x'
    label2 = 'libcryptoopenssl-OpenSSL_0_9_8t'
    #label3 is both, label4 is none
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    global_dict = p.load(open('iteration/max15','r'))
    for node in nodeList:
        nodename = node.get_name().strip('\"')
        if nodename in global_dict:
            labelset = global_dict[nodename]
            if label1 in labelset and label2 in labelset:
                count3 += 1
                node.obj_dict['attributes']['class']=3
            elif label1 in labelset:
                count1 += 1
                node.obj_dict['attributes']['class']=1
            elif label2 in labelset:
                count2 += 1
                node.obj_dict['attributes']['class']=2
            else:
                count4 += 1
                node.obj_dict['attributes']['class']=4
        else:
            count4 += 1
            node.obj_dict['attributes']['class']=4
    #print graph.to_string()
    print count1, count2, count3, count4

def addattributeforfunctions():
    path = '/home/yijiufly/Downloads/codesearch/data/mupdf/MUPDF-all/mupdf-1.15.0/mutool_bn.dot'
    graph = pydot.graph_from_dot_file(path)
    nodeList = graph.get_node_list()
    dir_freetype2 = '/home/yijiufly/Downloads/codesearch/data/mupdf/freetype2-test/VER-2-9-1/'
    nam = 'libfreetype.so_newmodel_withsize.nam'
    names = p.load(open(dir_freetype2+nam, 'r'))
    ft_name = [i[0] for i in names]
    dir_jpeg = '/home/yijiufly/Downloads/codesearch/data/mupdf/libjpeg-test/v9-pre/'
    nam = 'libjpeg.so_newmodel_withsize.nam'
    names = p.load(open(dir_jpeg+nam, 'r'))
    jpeg_name = [i[0] for i in names]
    import csv
    with open('/home/yijiufly/Downloads/codesearch/data/mupdf/MUPDF-all/mupdf-1.15.0/out_prediction1120_BP.csv', 'rb') as f:
        reader = csv.reader(f)
        prediction = list(reader)
    wrong_nodes = set()
    for line in prediction:
        if line[-1] == 'False':
            wrong_nodes.add(line[0])
    # correctname = []
    # for key in prediction:
    #     pred = [i[0][0] for i in prediction[key]]
    #     if key in pred:
    #         correctname.append(key)
    for node in nodeList:
        nodename = node.get_name().strip('\"')
        if nodename in ft_name:
            node.obj_dict['attributes']['class']=0
        elif nodename in jpeg_name:
            node.obj_dict['attributes']['class']=1
        else:
            node.obj_dict['attributes']['class']=2
        if nodename in wrong_nodes:
            node.obj_dict['attributes']['wrong_nodes']=1
        else:
            node.obj_dict['attributes']['wrong_nodes']=0
    print graph.to_string()

def removesomenode():
    path = '/home/yijiufly/Downloads/codesearch/BP.dot'
    graph = pydot.graph_from_dot_file(path)
    nodeList = graph.get_node_list()
    filtednames = p.load(open('../nodes_filted.p','r'))
    graph_string = 'digraph Call_Graph {\n'
    graph_string += ';\n'.join([rv for rv in filtednames])
    edgeList = graph.get_edge_list()
    edge_string = ''
    for edges in edgeList:
        if edges.get_source()=='main' or edges.get_destination()== 'main':
            edge_string += '{} -> {} [label={}];\n'.format(edges.get_source(), edges.get_destination(), edges.obj_dict['attributes']['label'])
    graph_string += ';\n' + edge_string
    graph_string += '\n}'
    print graph_string
addattributeforfunctions()
