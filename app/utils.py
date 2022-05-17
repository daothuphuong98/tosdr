import docx2txt
from PyPDF2 import PdfFileReader
import json

from flask import Response

class Methods:
    RF = 'Random Forest'
    SVC = 'SVC'
    LGBM = 'Light GBM'
    BERT = 'BERT'
    ROBERTA = 'ROBERTA'


def ok_json(data):
    return Response(json.dumps(data), status=200, mimetype='application/json')


def method_not_allowed(msg):
    return Response(json.dumps({'msg': msg}), status=405, mimetype='application/json')


def is_valid_method(method):
    if method in [Methods.RF, Methods.SVC, Methods.GB, Methods.ES]:
        return True
    return False

def read_pdf(file):
    try:
        pdfReader = PdfFileReader(file)
        count = pdfReader.numPages
        all_page_text = ""
        for i in range(count):
            page = pdfReader.getPage(i)
            all_page_text += page.extractText()
        return True, all_page_text
    except Exception as e:
        return False, str(e)

def read_docx(file):
    try:
        return True, docx2txt.process(file)
    except Exception as e:
        return False, e

def read_txt(file):
    try:
        return True, str(file.read(), "utf-8")
    except Exception as e:
        return False, e


