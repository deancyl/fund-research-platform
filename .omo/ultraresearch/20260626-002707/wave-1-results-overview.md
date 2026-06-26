# Wave 1-Results-Overview
## Wave 1: Saturation Research (11 axes)

### Repositories Cloned (5 total)
1. **mlfinlab** (hudson-and-thames) - Lopez de Prado AFML implementations (stubs: labeling, fracdiff, cross-validation, combinatorial CV, feature importance, ensemble)
2. **fracdiff** (SimonWard) - Working fractional differentiation implementation in Python (NumPy/SciPy backend + sklearn + torch)
3. **qlib** (microsoft) - AI-oriented quantitative investment platform with ensemble, meta-learning, online model
4. **hummingbot** - Production triple-barrier implementation in v2 strategy framework
5. **FinRL** (AI4Finance) - Deep RL for finance with ensemble strategy

### Key Source Files Found
- MLFinLab: labeling/labeling.py, features/fracdiff.py, cross_validation/cross_validation.py, cross_validation/combinatorial.py, feature_importance/importance.py, ensemble/sb_bagging.py, bet_sizing/bet_sizing.py, data_generation/corrgan.py, structural_breaks/cusum.py
- fracdiff: fdiff.py (actual working implementation)
- Qlib: model/ens/ensemble.py, model/meta/model.py, contrib/model/double_ensemble.py, contrib/online/online_model.py
- Hummingbot: position_executor with triple_barrier_config, TripleBarrierConfig class
- FinRL: ensemble_stock_trading.py (DRL ensemble with rebalancing)

### Papers Found
- CorrGAN: arXiv:1910.09504 - GAN for financial correlation matrices
