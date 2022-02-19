import sqlalchemy
import pandas as pd
from sqlalchemy import inspect, text, insert, MetaData

class MySQLDB:
    def __init__(self, host, user, passwd, db):
        self.host = host
        self.user = user
        self.pw = passwd
        self.db = db

        self.engine = sqlalchemy.create_engine(f'mysql+pymysql://{self.user}:{self.pw}@{self.host}/{self.db}')
        self.meta_data = MetaData()
        self.meta_data.reflect(bind=self.engine)
        conn = self.engine.connect()
        conn.close()

    def get_table_names(self):
        insp = inspect(self.engine)
        return insp.get_table_names()

    def insert(self, table, values):
        '''
        param table: Name of table to insert to
        param values: List of dictionaries, with each dict being a row and having keys as column name
                      and value as value of the column
        '''
        with self.engine.connect() as conn:
            table = self.meta_data.tables[table]
            conn.execute(insert(table), values)

    def insert_pandas(self, table, pd_dataframe):
        pd_dataframe.to_sql(table, self.engine, index=False)

    def query(self, sql_query=None):
        return pd.read_sql(sql_query, self.engine)

    def execute_sql(self, sql_statement):
        with self.engine.connect() as conn:
            result = conn.execute(text(sql_statement))
        return result

