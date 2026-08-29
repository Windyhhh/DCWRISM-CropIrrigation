<div align="center">

# 作物灌溉模型 | DCWRISM-CropIrrigation

### Physics-empirical crop irrigation water demand model.

FAO Penman-Monteith + differential-evolution optimization for crop irrigation water demand — 21.82% error on real data.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

</div>

---

**DCWRISM-CropIrrigation** is a crop irrigation water demand model that blends the FAO-56 Penman-Monteith reference evapotranspiration with an empirical crop coefficient, then tunes the parameters with differential evolution. The result is a light, reproducible model that predicts crop water demand with only **21.82%** error.

> [!NOTE]
> 中文项目：物理经验混合的智能灌溉需水模型，基于 FAO Penman-Monteith 与差分进化（DE）参数优化。

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/DCWRISM-CropIrrigation.git
cd DCWRISM-CropIrrigation

pip install -r requirements.txt

# Load data and run the base model
python dcwrism_model.py

# Advanced DE-optimized model
python FINAL_OPTIMIZED_HONEST_MODEL.py
```

---

## Features

- **FAO Penman-Monteith ET₀** — reference evapotranspiration from weather inputs.
- **Differential-evolution tuning** — auto-searches crop coefficients and soil factors.
- **Honest, reproducible** — final model with verified 21.82% error on real regional data.

---

## Project Structure

```
DCWRISM-CropIrrigation/
├── dcwrism_model.py              # core irrigation water demand model
├── data_loader.py                # load & parse weather / region data
├── ADVANCED_OPTIMIZATION_MODEL.py
├── FINAL_OPTIMIZED_HONEST_MODEL.py   # final DE-tuned model
└── requirements.txt
```

---

## 技术实现细节

### 架构概览

项目采用扁平结构，核心代码位于根目录。

### 核心类与模块

- **DataLoader**

### 关键函数

- `evaluate_model`, `evaluate_test_years`, `load_weather_data`, `load_crop_area`

### 技术栈与依赖

**核心框架/库**：NumPy, pandas

**主要 import**：
```python
import sys
import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
from data_loader import DataLoader
from dcwrism_model import DCWRISMModel
import io
import sys
import pandas as pd
import numpy as np
```

### 实现要点

- 以 `DataLoader` 为核心类，封装主要业务逻辑
- 通过 `evaluate_model` 等函数实现核心流程编排
- 基于 NumPy, pandas 构建，保证技术栈成熟稳定
- 代码结构清晰，模块间低耦合，便于扩展和维护

---
## License

MIT — free to use, modify and distribute.
