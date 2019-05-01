import binascii
import hashlib
import hmac
import io
import os
import pickle
import tempfile
import traceback

import pymongo
from bson import ObjectId
from flask import Flask, jsonify, request, abort, after_this_request

import embedding
import raw_graphs

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
windows = os.name == 'nt'
embedding.flags.DEFINE_string('bind', '', 'Server address')
embedding_engine = embedding.Embedding()
SECRET_KEY = binascii.unhexlify('8a5e271d56033e75f121719594cb1993c267d57013adc1bc07bc8672bbf726b7')
client = pymongo.MongoClient(
    os.environ.get('DATABASE_HOST', '10.8.0.1'),
    username=os.environ.get('DATABASE_USERNAME', 'admin'),
    password=os.environ.get('DATABASE_PASSWD', 'deepbits'),
    authSource='admin'
)
db = client[os.environ.get('DATABASE_NAME', 'coogledb')]
FILTER_SIZE = 10
TOP_K = 20
MIN_SCORE = 0.9
MAX_BASIC_BLOCK_NUM_DIFF = 0.4


@app.errorhandler(Exception)
def on_error(exception):
    traceback.print_exc()
    response = jsonify(error=str(exception))
    response.status_code = 500
    return response


def _compute_checksum(f):
    auth = hmac.new(SECRET_KEY, digestmod=hashlib.sha256)
    for chunk in iter(lambda: f.read(4096), b''):
        auth.update(chunk)
    f.seek(0)
    return auth.hexdigest()


def _compare(data, emb):
    if len(emb[1]) == 0:
        return []
    func_names = emb[0]
    for v in data:
        v['embedding'] = pickle.loads(v['embedding'])
    data = [v for v in data if len(v['embedding']) >= FILTER_SIZE]
    if len(data) == 0:
        return []
    vul_data = [v['embedding'] for v in data]
    preds = embedding_engine.test_similarity(vul_data, emb[1])
    ret = []
    for vul_idx in range(len(preds)):
        similarities = preds[vul_idx]
        idx = sorted(range(len(similarities)), key=lambda k: similarities[k], reverse=True)
        vul = data[vul_idx]

        functions = []
        for i in range(len(idx)):
            cur = idx[i]
            score = similarities[cur].item()
            if score < MIN_SCORE or len(functions) > TOP_K:
                break
            num_bb = len(emb[1][cur])
            if abs(num_bb - len(vul_data[vul_idx])) / num_bb > MAX_BASIC_BLOCK_NUM_DIFF:
                continue
            functions.append({'function': func_names[cur], 'score': score})
        if len(functions) == 0:
            continue
        ret.append({
            'vulId': str(vul['_id']),
            'name': vul['name'],
            'description': vul.get('description', ''),
            'type': vul.get('type', ''),
            'functionName': vul['functionName'],
            'functions': functions
        })
    return ret


@app.route('/', methods=['POST'])
def handle():
    request_file = request.files['file']
    method = request.form['method']
    computed_checksum = _compute_checksum(request_file)
    checksum = request.headers.get('Checksum', '')
    if not hmac.compare_digest(checksum, computed_checksum):
        abort(400)
    if method == 'embedding':
        with tempfile.NamedTemporaryFile(delete=not windows) as f:
            @after_this_request
            def clean_up(resp):
                if windows:
                    os.remove(f.name)
                return resp

            request_file.save(f.name)
            emb = embedding_engine.embed_a_binary(f.name)
            if 'funcName' in request.form:
                func_name = request.form['funcName']
                idx = emb[0].index(func_name)
                emb = emb[1][idx]
            emb = pickle.dumps(emb)
            emb_len = len(emb)
            emb = io.BytesIO(emb)
            auth = _compute_checksum(emb)
            return app.response_class(emb, mimetype='application/octet-stream', headers={
                'Content-Length': emb_len,
                'Checksum': auth,
            })
    elif method == 'compare':
        filter = [{'productId': None}]
        if 'productId' in request.form:
            filter.append({'productId': ObjectId(request.form['productId'])})
        data = list(db.vulnerabilities.find({'$or': filter}))
        emb = pickle.load(request_file)
        ret = _compare(data, emb)
        return jsonify(ret)
    abort(400)


if __name__ == '__main__':
    app.run()
