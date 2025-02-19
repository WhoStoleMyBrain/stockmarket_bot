from typing import Any
class CoinbaseProduct:
    def __init__(self, api_response: dict[str: Any]):
        self.product_id = self.use_if_exists_else_empty(api_response, "product_id")
        self.price = self.use_if_exists_else_0(api_response, "price")
        self.price_percentage_change_24h = self.use_if_exists_else_0(api_response, "price_percentage_change_24h")
        self.volume_24h = self.use_if_exists_else_0(api_response, "volume_24h")
        self.volume_percentage_change_24h = self.use_if_exists_else_0(api_response, "volume_percentage_change_24h")
        self.base_increment = self.use_if_exists_else_0(api_response, "base_increment")
        self.quote_increment = self.use_if_exists_else_0(api_response, "quote_increment")
        self.quote_min_size = self.use_if_exists_else_0(api_response, "quote_min_size")
        self.quote_max_size = self.use_if_exists_else_0(api_response, "quote_max_size")
        self.base_min_size = self.use_if_exists_else_0(api_response, "base_min_size")
        self.base_max_size = self.use_if_exists_else_0(api_response, "base_max_size")
        self.base_name = self.use_if_exists_else_empty(api_response, "base_name")
        self.quote_name = self.use_if_exists_else_empty(api_response, "quote_name")
        self.watched = self.use_if_exists_else_false(api_response, "watched")
        self.is_disabled = self.use_if_exists_else_false(api_response, "is_disabled")
        self.new = self.use_if_exists_else_false(api_response, "new")
        self.status = self.use_if_exists_else_empty(api_response, "status")
        self.cancel_only = self.use_if_exists_else_false(api_response, "cancel_only")
        self.limit_only = self.use_if_exists_else_false(api_response, "limit_only")
        self.post_only = self.use_if_exists_else_false(api_response, "post_only")
        self.trading_disabled = self.use_if_exists_else_false(api_response, "trading_disabled")
        self.auction_mode = self.use_if_exists_else_false(api_response, "auction_mode")
        self.product_type = self.use_if_exists_else_empty(api_response, "product_type")
        self.quote_currency_id = self.use_if_exists_else_0(api_response, "quote_currency_id")
        self.base_currency_id = self.use_if_exists_else_0(api_response, "base_currency_id")
        self.fcm_trading_session_details = self.use_if_exists_else_empty(api_response, "fcm_trading_session_details")
        self.mid_market_price = self.use_if_exists_else_0(api_response, "mid_market_price")
        self.alias = self.use_if_exists_else_empty(api_response, "alias")
        self.alias_to = self.use_if_exists_else_empty(api_response, "alias_to")
        self.base_display_symbol = self.use_if_exists_else_empty(api_response, "base_display_symbol")
        self.quote_display_symbol = self.use_if_exists_else_empty(api_response, "quote_display_symbol")
        self.view_only = self.use_if_exists_else_false(api_response, "view_only")
        self.price_increment = self.use_if_exists_else_0(api_response, "price_increment")
        self.display_name = self.use_if_exists_else_empty(api_response, "display_name")
        self.product_venue = self.use_if_exists_else_empty(api_response, "product_venue")
        self.approximate_quote_24h_volume = self.use_if_exists_else_0(api_response, "approximate_quote_24h_volume")
        self.new_at = self.use_if_exists_else_empty(api_response, "new_at")
    
    def use_if_exists_else_0(self, api_response: dict[str: Any], key: str) -> float:
        if key in api_response.keys():
            try:
                return float(api_response[key])
            except:
                return 0.0
        return 0.0
    
    def use_if_exists_else_empty(self, api_response: dict[str: Any], key: str) -> str:
        if key in api_response.keys():
            return api_response[key]
        return ""
    
    def use_if_exists_else_false(self, api_response: dict[str: Any], key: str) -> bool:
        if key in api_response.keys():
            try:
                return bool(api_response[key])
            except:
                return False
        return False
    
    def __str__(self):
        return f"{self.product_id}:{self.base_increment}:{self.quote_increment}"