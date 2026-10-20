# Research Log

## Oct 20, 2026 - Code Audit & Bug Fixes

### Critical Issues Found
- Genuine score extraction using first match instead of max
- Sentinel values (-1.0) contaminating statistics
- ROC curve computation bias
- Silent image loading failures
- Missing error validation

### Fixes Applied
- Changed genuine score to use `.max()` 
- Filtered sentinel values before all calculations
- Fixed ROC computation with clean data
- Added logging and validation
- Fixed file handle leaks
- Added directory existence checks

### Testing
- 8/8 unit tests pass
- 4 models evaluated
- Accuracy verified at 98.63%

### Status
✅ All fixes validated and committed
✅ Ready for submission
