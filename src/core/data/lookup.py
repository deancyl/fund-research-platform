"""
Fund lookup engine — fuzzy search by code, Chinese name, or pinyin.
"""


class FundLookupEngine:
    """In-memory fund registry with fuzzy matching."""

    def __init__(self) -> None:
        self._registry: list[dict[str, str]] = [
            {"code": "000001", "name": "华夏成长混合", "pinyin": "huaxiachengzhang"},
            {"code": "005827", "name": "易方达蓝筹精选混合", "pinyin": "yifangdalanchou"},
            {"code": "510300", "name": "华泰柏瑞沪深300ETF", "pinyin": "huataibairuihushen300etf"},
            {"code": "159915", "name": "易方达创业板ETF", "pinyin": "yifangdachuangyebanetf"},
            {"code": "161725", "name": "招商中证白酒指数", "pinyin": "zhaoshangzhongzhengbai"},
            {"code": "510880", "name": "华泰柏瑞红利ETF", "pinyin": "huataibairuihonglietf"},
        ]

    def search(self, query: str) -> list[dict[str, str]]:
        """Fuzzy search by code prefix, name substring, or pinyin."""
        if not query:
            return []
        q = query.strip().lower()
        results: list[dict[str, str]] = []
        for item in self._registry:
            if q in item["code"] or q in item["name"].lower() or q in item["pinyin"]:
                results.append(item)
        return results
