
import pandas as pd
from sqlalchemy import create_engine

host = 'prod-pentaho.cxfaihg8elfv.eu-west-1.rds.amazonaws.com'
db = 'billin_prod'
user = 'billin'
password = 'ThisIsTheRiverOfTheNight'






class DB():

    def __init__(self, user=None, password=None, host=None, db=None):

        #self.user = user
        #self.password = password
        #self.host = host
        #self.db = db
        self.engine = create_engine('postgresql://{}:{}@{}:5432/{}'.format(user, password,host,db))


    def gettable(self, db):
        #contacts = pd.read_sql_query('select * from "contacts"',con=engine)
        #users = pd.read_sql_query('select * from "users"',con=engine)
        #sessions = pd.read_sql_query('select * from "sessions"',con=engine)
        #premiums = pd.read_sql_query('select * from "premiums"',con=engine)
        #gocardlesses = pd.read_sql_query('select * from "gocardlesses"',con=engine)
        #campaigns = pd.read_sql_query('select * from "campaigns"',con=engine)
        #campaign_details = pd.read_sql_query('select * from "campaign-details"',con=engine)
        #businessesUsers = pd.read_sql_query('select * from "businessesUsers"',con=engine)
        #businesses = pd.read_sql_query('select * from "businesses"',con=engine)
        #businessConstacts = pd.read_sql_query('select * from "businessContacts"',con=engine)
        #bankAccounts = pd.read_sql_query('select * from "bankAccounts"',con=engine)
        #addresses = pd.read_sql_query('select * from "addresses"',con=engine)

        return pd.read_sql_query('select * from "{}"'.format(db),con=self.engine)

