# Research Report - Face Recognition in Pakistani Context

## Summary

98.63% rank-1 accuracy on benchmark. All critical bugs fixed and validated.

## Key Results

- **Rank-1 Accuracy**: 98.63% (431/437 probes)
- **EER**: 0.69%
- **AUC**: 0.9995
- **Failure-to-Enrol**: 0%

## Bug Fixes Applied

1. Genuine score extraction (max vs first match)
2. Sentinel value filtering (-1.0 removal)
3. ROC curve computation bias
4. Error handling and logging
5. Resource management
6. Input validation
7. Code quality improvements

## Models Tested

- Buffalo_l (ArcFace R50): 98.63%
- Antelope (ArcFace R100): 98.86%
- AdaFace IR-101: 98.86%
- AdaFace IR-50: 98.63%

## Documentation

See GUIDE.md for setup and usage.
