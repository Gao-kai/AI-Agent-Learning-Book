max_order_count = 1000000


def create_order():
    print("这是创建订单")


def cancel_order():
    print("这是删除订单")


def show():
    print(f"这是【订单】模块")


# 限制其他模块通过from order import * 时的可导出对象
__all__ = ["create_order", "cancel_order"]
