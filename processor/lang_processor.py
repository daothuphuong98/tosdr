from processor import Processor
import langid

class LanguageProcessor(Processor):
    def __init__(self):
        super().__init__()

    def process(self, entry):
        if entry is None:
            return entry
        short_text = entry[:2000]
        lang = langid.classify(short_text)[0]
        if lang == 'en':
            return entry
        return None

