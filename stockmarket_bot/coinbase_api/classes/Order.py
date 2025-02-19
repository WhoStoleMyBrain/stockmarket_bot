from typing import Any
class Order:
    def __init__(self, order:dict[str: Any]):
        #! needed for inducing bracket sell order
        self.order_side = order["order_side"]
        self.status = order["status"]
        self.avg_price = order["avg_price"]
        self.cumulative_quantity = order["cumulative_quantity"]
        self.total_value_after_fees = order["total_value_after_fees"]
        self.product_id = order["product_id"]
        #! next two are just for sanity check
        self.completion_percentage = order["completion_percentage"]
        self.leaves_quantity = order["leaves_quantity"]
        
    def is_finished_buy_order(self) -> bool:
        # take each condition step by step
        if self.order_side == "BUY":
            print("order is buy side... check!")
            if self.status == "FILLED":
                print("status is FILLED... check!")
                if float(self.completion_percentage) >= 100:
                    print("completion_percentage is 100... check!")
                    if float(self.leaves_quantity) >= 0:
                        print("leaves_quantity is 0... check!")
                        return True
        return False