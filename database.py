from pymongo import MongoClient
from config import MONGODB_URI, MONGODB_DB_NAME

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

def get_collection(collection_name: str):
    return db[collection_name]
