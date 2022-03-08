import logging

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s', level=logging.ERROR)


class Processor(object):

    def __init__(self):
        self.num_processed = 0
        self.num_skip = 0
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.name)
        self.log.setLevel(logging.INFO)

    def process(self, entry):
        raise NotImplementedError

    def stats(self):
        """return statistics of processor
        """
        return {
            'num_processed': self.num_processed,
            'num_skip': self.num_skip
        }