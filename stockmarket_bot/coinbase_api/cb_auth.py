import http.client
import hmac
import hashlib
import json
import time
from urllib.parse import urlencode
from typing import Union, Dict
import time


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

    def set_credentials(self, api_key, api_secret):
        """
        Update the API credentials used for authentication.
        :param api_key: The API Key for Coinbase API
        :param api_secret: The API Secret for Coinbase API
        """
        self.key = api_key
        self.secret = api_secret

    def __call__(self, method: str, path: str, body: Union[Dict, str] = '', params: Dict[str, str] = None) -> Dict:
        """
        Prepare and send an authenticated request to the Coinbase API.

        :param method: HTTP method (e.g., 'GET', 'POST')
        :param path: API endpoint path
        :param body: Request payload
        :param params: URL parameters
        :return: Response from the Coinbase API as a dictionary
        """
        path = self.add_query_params(path, params)
        body_encoded = self.prepare_body(body)
        headers = self.create_headers(method, path, body)
        return self.send_request(method, path, body_encoded, headers)

    def add_query_params(self, path, params):
        if params:
            query_params = urlencode(params)
            path = f'{path}?{query_params}'
        return path

    def prepare_body(self, body):
        return json.dumps(body).encode('utf-8') if body else b''

    def create_headers(self, method, path, body):
        timestamp = str(int(time.time()))
        message = timestamp + method.upper() + \
            path.split('?')[0] + (json.dumps(body) if body else '')
        signature = hmac.new(self.secret.encode(
            'utf-8'), message.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()

        return {
            "Content-Type": "application/json",
            "CB-ACCESS-KEY": self.key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp
        }

    def send_request(self, method, path, body_encoded, headers):
        retries = 0
        backoff_time = 1  # Initial backoff time in seconds

        while retries <= self.MAX_502_RETRIES:
            conn = http.client.HTTPSConnection("api.coinbase.com")

            response_data, status_code = self._execute_request(conn, method, path, body_encoded, headers)
            
            if status_code in self.ERROR_MAPPING:
                error_info = self.ERROR_MAPPING[status_code]
                print(f"Error {status_code}: {error_info['message']}")
                
                if error_info["action"] == "raise":
                    return {'errors': {error_info['message']}, 'status': status_code, 'data': response_data}
                elif error_info["action"] == "log":
                    return {'errors': {error_info['message']}, 'status': status_code, 'data': response_data}
                elif error_info["action"] == "retry":
                    retries += 1
                    if retries > self.MAX_502_RETRIES:
                        return {'errors': {f"{error_info['message']} After maximum retries"}, 'status': status_code, 'data': response_data}
                    time.sleep(backoff_time)  # Waiting before the next retry
                    backoff_time *= 2  # Double the backoff time for the next potential retry
                    continue
                
            return response_data
        
    def _execute_request(self, conn, method, path, body_encoded, headers):
        try:
            conn.request(method, path, body_encoded, headers)
            res = conn.getresponse()
            data = res.read()

            # Handle 502 Bad Gateway error as a special case
            if res.status == 502:
                print("Received a 502 Bad Gateway response.")
                return {'errors': {'message': 'Bad Gateway error!'}, 'status': 502, 'data': 'Bad Gateway error received'}, 502
            
            # If unable to decode JSON response, handle as another special case.
            try:
                response_data = json.loads(data.decode("utf-8"))
                return response_data, res.status
            except json.JSONDecodeError:
                print("Error: Unable to decode JSON response. Raw response data:", data)
                return {'errors': {'message': 'Unable to decode JSON response.'}, 'status': 400, 'data': data.decode("utf-8")}, 400
        finally:
            conn.close()