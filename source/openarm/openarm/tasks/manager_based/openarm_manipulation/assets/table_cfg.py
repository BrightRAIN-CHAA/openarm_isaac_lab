import os

ASSET_DIR = os.path.dirname(__file__)
TABLE_USD_PATH = os.path.join(ASSET_DIR, "SM_HeavyDutyPackingTable_C02_01", "SM_HeavyDutyPackingTable_C02_01_physics.usd")

print(f"내가 찾은 테이블 주소: {TABLE_USD_PATH}")