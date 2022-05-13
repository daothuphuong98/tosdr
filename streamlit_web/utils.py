import docx2txt
from PyPDF2 import PdfFileReader

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


