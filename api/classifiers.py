import os

from flasgger import Swagger
from flask import Flask, request

from api.service import TOSService
from api.utils import ok_json, is_valid_method, method_not_allowed, Methods

app = Flask(__name__)
app.config['SWAGGER'] = {
    'doc_dir': 'api/apidocs',
    "title": "Term of Service Demo API",
    "uiversion": 3,
}

swagger = Swagger(app)

tosservice = TOSService()


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Terms of Service Detection</h1>
    <p>This site is a prototype API for Terms of Service Detection.</p>'''


@app.route("/sentence", methods=["POST"])
def sentence():
    sentence = request.json['sentence']
    method = request.args.get('method', Methods.RF)
    if not is_valid_method(method):
        return method_not_allowed('Wrong method %s' % method)

    result = tosservice.detect(sentence, method)
    return ok_json(result)

@app.route("/paragraph", methods=["POST"])
def paragraph():
    paragraph = request.json['paragraph']
    method = request.args.get('method', Methods.RF)
    if not is_valid_method(method):
        return method_not_allowed('Wrong method %s' % method)

    result = tosservice.detect_paragraph(paragraph, method)
    return ok_json(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(port=port, debug=True)
