import traceback
try:
    import idc

    import os
    import hashlib
    import sys
    # idapython plugin used for extract callgraph
    if __name__ == '__main__':
        # path = '.'
        path = idc.ARGV[1] if len(idc.ARGV) >= 1 else '.'
        with open('/home/yijiufly/Downloads/codesearch/generateidaexception', 'w') as f:
            f.write(path)
        #path = '/home/yijiufly/Downloads/codesearch/a.gdl'
        # used for docker
        # path = "/output"
        idc.GenCallGdl(path, 'Call Gdl', idc.CHART_GEN_GDL)
        idc.Message('Gdl file has been saved to {}\n'.format(path))
        idc.qexit(0)
except Exception:
    with open('/home/yijiufly/Downloads/codesearch/generateidaexception', 'a') as f:
        f.write(traceback.format_exc())
