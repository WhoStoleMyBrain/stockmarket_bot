
def get_json_1():
    return {
                        "avg_price": "3.394",
                        "cancel_reason": "",
                        "client_order_id": "eb33b532-771a-4432-b443-16bdc35c08c1",
                        "completion_percentage": "100.00",
                        "contract_expiry_type": "UNKNOWN_CONTRACT_EXPIRY_TYPE",
                        "cumulative_quantity": "0.746",
                        "filled_value": "2.531924",
                        "leaves_quantity": "0",
                        "limit_price": "3.394",
                        "number_of_fills": "1",
                        "order_id": "7aafde8c-7cf4-4939-aff2-3fc5f42f4863",
                        "order_side": "SELL",
                        "order_type": "Limit",
                        "outstanding_hold_amount": "0",
                        "post_only": "false",
                        "product_id": "NEAR-USDC",
                        "product_type": "SPOT",
                        "reject_Reason": "",
                        "retail_portfolio_id": "2cc964c2-2f11-518c-80c7-acdaa56f2170",
                        "risk_managed_by": "UNKNOWN_RISK_MANAGEMENT_TYPE",
                        "status": "FILLED",
                        "stop_price": "",
                        "time_in_force": "GOOD_UNTIL_CANCELLED",
                        "total_fees": "0.008861734",
                        "total_value_after_fees": "2.523062266",
                        "trigger_status": "INVALID_ORDER_TYPE",
                        "creation_time": "2025-02-16T08:28:35.373422Z",
                        "end_time": "0001-01-01T00:00:00Z",
                        "start_time": "0001-01-01T00:00:00Z"
                    }
    
def get_json_2():
    return {
                        "avg_price": "3.394",
                        "cancel_reason": "",
                        "client_order_id": "eb33b532-771a-4432-b443-16bdc35c08c1",
                        "completion_percentage": "100.00",
                        "contract_expiry_type": "UNKNOWN_CONTRACT_EXPIRY_TYPE",
                        "cumulative_quantity": "0.746",
                        "filled_value": "2.531924",
                        "leaves_quantity": "0",
                        "limit_price": "3.394",
                        "number_of_fills": "1",
                        "order_id": "7aafde8c-7cf4-4939-aff2-3fc5f42f4863",
                        "order_side": "SELL",
                        "order_type": "Limit",
                        "outstanding_hold_amount": "0.746",
                        "post_only": "false",
                        "product_id": "NEAR-USDC",
                        "product_type": "SPOT",
                        "reject_Reason": "",
                        "retail_portfolio_id": "2cc964c2-2f11-518c-80c7-acdaa56f2170",
                        "risk_managed_by": "UNKNOWN_RISK_MANAGEMENT_TYPE",
                        "status": "FILLED",
                        "stop_price": "",
                        "time_in_force": "GOOD_UNTIL_CANCELLED",
                        "total_fees": "0.008861734",
                        "total_value_after_fees": "2.523062266",
                        "trigger_status": "INVALID_ORDER_TYPE",
                        "creation_time": "2025-02-16T08:28:35.373422Z",
                        "end_time": "0001-01-01T00:00:00Z",
                        "start_time": "0001-01-01T00:00:00Z"
                    }

def main():
    json_1 = get_json_1()
    json_2 = get_json_2()
    json_2_keys = json_2.keys()
    print(f"second json keys: {json_2_keys}")
    for key in json_1.keys():
        if key in json_2_keys:
            print(f"{key}:\n\t{json_1[key]}\n\t{json_2[key]}")
        else:
            print(f"key {key} was not in keys of second json.")
    return

if __name__ == '__main__':
    main()