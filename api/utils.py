import json

from flask import Response


class Methods:
    RF = 'Random Forest'
    SVC = 'SVC'
    GB = 'Gradient Boosting'
    ES = 'Ensemble'

def ok_json(data):
    return Response(json.dumps(data), status=200, mimetype='application/json')


def method_not_allowed(msg):
    return Response(json.dumps({'msg': msg}), status=405, mimetype='application/json')


def is_valid_method(method):
    if method in [Methods.RF, Methods.SVC, Methods.GB, Methods.ES]:
        return True
    return False