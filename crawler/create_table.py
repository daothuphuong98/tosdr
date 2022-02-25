from utils.sqldb import MySQLDB

if __name__ == '__main__':
    dem = MySQLDB('34.131.19.243', 'user', 'tosdr2022','raw')
    # dem.execute_sql('''
    #                     CREATE TABLE IF NOT EXISTS SERVICE
    #                     (
    #                         SERVICE_ID VARCHAR(10) PRIMARY KEY,
    #                         NAME VARCHAR(255),
    #                         RATING VARCHAR(10),
    #                         LINK VARCHAR(50)
    #                     )
    #                  '''
    #                 )
    #
    # dem.execute_sql('''
    #                     CREATE TABLE IF NOT EXISTS TOPIC
    #                     (
    #                         TOPIC_ID VARCHAR(10) PRIMARY KEY,
    #                         NAME VARCHAR(255),
    #                         LINK VARCHAR(50)
    #                     )
    #                  '''
    #                 )
    #
    # dem.execute_sql('''
    #                     CREATE TABLE IF NOT EXISTS DOCUMENT
    #                     (
    #                         DOC_ID VARCHAR(10) PRIMARY KEY,
    #                         SERVICE_ID VARCHAR(10) REFERENCES SERVICE(SERVICE_ID),
    #                         TEXT MEDIUMTEXT,
    #                         TYPE VARCHAR(255),
    #                         LINK VARCHAR(50)
    #                     )
    #                  '''
    #                 )
    #
    # dem.execute_sql('''
    #                     CREATE TABLE IF NOT EXISTS CASES
    #                     (
    #                         CASE_ID VARCHAR(10) PRIMARY KEY,
    #                         LINK VARCHAR(50),
    #                         TOPIC_ID VARCHAR(10) REFERENCES TOPIC(TOPIC_ID),
    #                         TITLE VARCHAR(500),
    #                         DESCRIPTION VARCHAR(1000),
    #                         RATING VARCHAR(10)
    #                     )
    #                  '''
    #                 )
    #
    # dem.execute_sql('''
    #                     CREATE TABLE IF NOT EXISTS POINT
    #                     (
    #                         POINT_ID VARCHAR(10) PRIMARY KEY,
    #                         LINK VARCHAR(50),
    #                         STATUS VARCHAR(10),
    #                         CASE_ID VARCHAR(10) REFERENCES CASES(CASE_ID),
    #                         SERVICE_ID VARCHAR(10) REFERENCES SERVICE(SERVICE_ID),
    #                         DOC_ID VARCHAR(10) REFERENCES DOCUMENT(DOC_ID),
    #                         SOURCE VARCHAR(1000),
    #                         TEXT TEXT
    #
    #                     )
    #                  '''
    #                 )
    # print(dem.get_table_names())
