import http.client
import hmac
import hashlib
import json
import time
from urllib.parse import urlencode
from typing import Union, Dict
from coinbase.rest import RESTClient
from coinbase import jwt_generator


class CBAuth:
    """
    Singleton class for Coinbase authentication.
    """

    MAX_502_RETRIES = 3

    ERROR_MAPPING = {
        401: {"message": "Unauthorized. Please check your API key and secret.", "action": "raise"},
        400: {"message": "Bad request. Went horribly wrong!", "action": "log"},
        502: {"message": "Bad Gateway error!", "action": "retry"}
    }

    _instance = None  # Class attribute to hold the singleton instance

    restClientInstance = RESTClient()

    def __new__(cls):
        """
        Override the __new__ method to control the object creation process.
        :return: A single instance of CBAuth
        """
        if cls._instance is None:
            print("Creating CBAuth instance")
            cls._instance = super(CBAuth, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        """
        Initialize the CBAuth instance with API credentials.
        """
        self.key = None
        self.secret = None
        self._restClientInstance = None

    def set_credentials(self, api_key, api_secret):
        """
        Update the API credentials used for authentication.
        :param api_key: The API Key for Coinbase API
        :param api_secret: The API Secret for Coinbase API
        """
        self.key = api_key
        self.secret = api_secret
        self.restClientInstance = RESTClient(api_key=self.key, api_secret=self.secret)

    