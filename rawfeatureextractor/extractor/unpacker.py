import tarfile
import json
import io
import os
import hashlib


def find_layers(img, id):
    layers = []
    while True:
        layers.append(id)
        with img.extractfile('%s/json' % id) as fd:
            info = json.load(io.TextIOWrapper(fd, 'utf-8'))
        if 'parent' in info:
            id = info['parent']
        else:
            return reversed(layers)


def unpack(img_path, output):
    with tarfile.open(img_path) as img:
        with img.extractfile('repositories') as repos:
            repos_text = json.loads(repos.read().decode('utf-8'))
        if len(repos_text) != 1:
            print('error: multiple images\n')
            sys.exit(0)
        latest = list(list(repos_text.values())[0].values())[0]
        layers = list(find_layers(img, latest))

        file_info = []
        for id in layers:
            with tarfile.open(fileobj=img.extractfile('%s/layer.tar' % id)) as layer:
                for member in layer.getmembers():
                    base = os.path.basename(member.path)
                    path = os.path.join(output, member.path)
                    dirname = os.path.dirname(path)
                    if base.startswith('.wh.'):
                        try:
                            os.unlink(os.path.join(dirname, base[4:]))
                        except OSError as e:
                            print(e)
                        continue

                    if not member.isfile():
                        continue

                    try:
                        layer.extract(member, output, False)
                    except OSError as e:
                        print(e)
                        continue

                    with open(path, 'rb') as fd:
                        sha256 = hashlib.sha256()
                        for block in iter(lambda: fd.read(65536), b''):
                            sha256.update(block)

                    file_info.append({
                        'path': os.path.dirname(member.path),
                        'name': base,
                        'sha256': sha256.hexdigest()
                    })

        with open(os.path.join(output, 'file_info.json'), 'w', encoding='utf-8') as fd:
            json.dump(file_info, fd)
