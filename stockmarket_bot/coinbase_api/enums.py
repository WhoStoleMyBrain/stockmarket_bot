from enum import Enum

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

class Method(Enum):
    POST = "POST"
    GET = "GET"

class Granularities(Enum):
    UNKNOWN_GRANULARITY = 'UNKNOWN_GRANULARITY'
    ONE_MINUTE = 'ONE_MINUTE'
    FIVE_MINUTE = 'FIVE_MINUTE'
    FIFTEEN_MINUTE = 'FIFTEEN_MINUTE'
    THIRTY_MINUTE = 'THIRTY_MINUTE'
    ONE_HOUR = 'ONE_HOUR'
    TWO_HOUR = 'TWO_HOUR'
    SIX_HOUR = 'SIX_HOUR'
    ONE_DAY = 'ONE_DAY'

class OrderStatus(Enum):
    OPEN = 'OPEN'
    CANCELLED = 'CANCELLED'
    EXPIRED = 'EXPIRED'

class OrderType(Enum):
    UNKNOWN_ORDER_TYPE = 'UNKOWN_ORDER_TYPE'
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    STOP = 'STOP'
    STOP_LIMIT = 'STOP_LIMIT'

class ProductType(Enum):
    SPOT = 'SPOT'
    FUTURE = 'FUTURE'

class Database(Enum):
    DEFAULT = 'default'
    HISTORICAL = 'historical'
    SIMULATION = 'simulation'

class Actions(Enum):
    SELL = 0
    HOLD = 1
    BUY = 2
    
class ExportFolder(Enum):
    EXPORT_FOLDER = "crypto_data_export"
    
class ApiPath(Enum):
    ORDERS_PREVIEW = "/api/v3/brokerage/orders/preview"
    ORDERS = "/api/v3/brokerage/orders"
    PRODUCTS = "/api/v3/brokerage/products"
    ACCOUNTS = "/api/v3/brokerage/accounts"
    BEST_BID_ASKS = "/api/v3/brokerage/best_bid_ask"
    