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

## License

MIT — free to use, modify and distribute.
