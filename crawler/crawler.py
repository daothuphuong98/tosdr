import logging
from utils.sqldb import MySQLDB
import yaml
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s', level=logging.INFO)

class BasicCrawler:
    def __init__(self, db, webdriver_init=True, config_file='resources/config.yaml'):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)
        self.log.info('Init %s' % self.__class__.__name__)

        if webdriver_init:
            options = Options()
            options.add_argument("--lang=en-US")
            options.add_argument("--incognito")
            options.add_argument("--log-level=2")
            self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

        with open(config_file) as fp:
            self.config = yaml.load(fp, Loader=yaml.FullLoader)

        self.db = MySQLDB(self.config['host'], self.config['user'], self.config['passwd'], db)

    def login(self):
        self.driver.get("https://edit.tosdr.org/users/sign_in")
        time.sleep(1)
        username = self.driver.find_element(By.CSS_SELECTOR, ".form-control.email")
        username.clear()
        username.send_keys(self.config['tosdr_mail'])

        password = self.driver.find_element(By.CSS_SELECTOR, ".form-control.password")
        password.clear()
        password.send_keys(self.config['tosdr_passwd'])
        password.send_keys(Keys.ENTER)
        time.sleep(2)
        if self.driver.current_url == 'https://edit.tosdr.org/':
            return True
        return False

    def crawl(self):
        pass

    def batch_insert(self):
        pass
