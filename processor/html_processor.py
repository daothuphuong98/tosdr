from bs4 import BeautifulSoup
from processor import Processor
import re

class HTMLProcessor(Processor):
    def __init__(self):
        super().__init__()

    def process(self, entry):
        if entry is None:
            return entry
        cleaned_html = BeautifulSoup(entry, "lxml").text
        cleaned_html = cleaned_html.replace(u'\xa0', u' ')
        return cleaned_html
