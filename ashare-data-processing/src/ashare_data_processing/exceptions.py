"""项目自定义异常。"""


class QuantError(Exception):
    """量化研究框架基础异常。"""


class ConfigurationError(QuantError):
    """配置缺失或配置值非法。"""


class DataValidationError(QuantError):
    """输入数据未满足标准字段或质量要求。"""


class ProviderError(QuantError):
    """外部数据源访问失败。"""

