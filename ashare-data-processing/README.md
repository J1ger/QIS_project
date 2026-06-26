# AShare Data Processing

本仓库是 A 股量化研究项目中“第一阶段：数据基础体系搭建”的代码交付部分，只保留数据处理相关模块，不包含因子研究、策略回测、绩效结果或可视化看板。

## 1. 项目内容

本模块基于 AkShare 构建 A 股日频数据处理框架，主要功能包括：

- AkShare 公开数据接口接入；
- A 股行情、财务、行业、北向资金、宏观 PMI 等字段的标准化整理；
- 本地 CSV 数据仓库；
- 按 `date/symbol` 主键进行增量更新和去重；
- 数据集 manifest 与 SHA256 哈希记录；
- 原始数据覆盖度、缺失率、异常值、停牌记录、重复键等质量评估；
- 历史成分股文件格式支持，用于降低幸存者偏差；
- 命令行入口和基础单元测试。

## 2. 目录结构

```text
ashare-data-processing/
├── config/
│   ├── akshare_example.json
│   └── historical_membership_example.csv
├── src/
│   └── ashare_data_processing/
│       ├── cli.py
│       ├── exceptions.py
│       ├── providers.py
│       ├── quality.py
│       └── storage.py
├── tests/
│   └── test_data_processing.py
├── environment.yml
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## 3. 环境安装

方式一：使用 pip。

```bash
pip install -r requirements.txt
pip install -e .
```

方式二：使用 conda。

```bash
conda env create -f environment.yml
conda activate ashare-data-processing
pip install -e .
```

## 4. 依赖版本

核心依赖已在 `requirements.txt` 和 `environment.yml` 中固定版本，主要包括：

| 依赖库 | 版本 | 用途 |
|---|---:|---|
| Python | 3.10 | 运行环境 |
| akshare | 1.18.64 | A 股公开数据接口 |
| numpy | 1.26.4 | 数值计算 |
| pandas | 2.2.2 | 表格数据处理 |
| pytest | 7.4.4 | 测试工具 |
| ruff | 0.5.0 | 代码检查工具 |
| setuptools | 69.5.1 | Python 包构建 |

## 5. 快速运行

```bash
ashare-data-processing --config config/akshare_example.json
```

运行后，程序会根据配置拉取样例数据，将数据保存到本地 `data/` 目录，并在终端输出数据质量评估结果。

## 6. 历史成分股文件格式

历史成分股文件用于标记股票在研究股票池中的有效区间，格式如下：

```csv
symbol,effective_from,effective_to,name
000001.SZ,2020-01-01,2021-06-30,平安银行
600000.SH,2020-01-01,,浦发银行
```

字段说明：

- `symbol`：标准股票代码；
- `effective_from`：进入样本池日期；
- `effective_to`：调出样本池日期，若仍在样本池中或暂无调出数据可留空；
- `name`：股票名称。

该机制可以降低幸存者偏差，但是否能够完全控制偏差，取决于是否具备完整的历史调入和调出记录。

## 7. 测试

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖：

- 合成数据质量评估；
- CSV 数据仓库增量更新；
- 交易日历补齐不生成上市前记录。

## 8. 不包含的内容

本仓库只提交数据处理层代码，不包含：

- 多因子计算和因子有效性检验；
- 策略回测和交易模拟；
- Dashboard 可视化；
- 本地缓存数据；
- 回测收益结果；
- 课程报告或 Word/PDF 文档。
