# Modlee Tests

## End-to-end testing

in root directory

### Dummy data

```
python3 -m pytest tests/end2end_tests/test_tabular_classification.py -k test
python3 -m pytest tests/end2end_tests/test_timeseries_forecasting.py -k test
python3 -m pytest tests/end2end_tests/test_image_classification.py -k test
python3 -m pytest tests/end2end_tests/test_image_segmentation.py -k test
python3 -m pytest tests/end2end_tests/test_tabular_regression.py -k test
python3 -m pytest tests/end2end_tests/test_image_regression.py -k test
python3 -m pytest tests/end2end_tests/test_timeseries_classification.py -k test
python3 -m pytest tests/end2end_tests/test_timeseries_regression.py -k test
python3 -m pytest tests/end2end_tests/test_image_to_image.py -k test
python3 -m pytest tests/end2end_tests/test_image_to_image_real_data_extended.py -k test

python3 -m pytest tests/end2end_tests/test_text_classification.py -k test
python3 -m pytest tests/end2end_tests/test_text_regression.py -k test
python3 -m pytest tests/end2end_tests/test_text_to_text.py -k test

python3 -m pytest tests/end2end_tests/test_text_to_text_single_input.py -k test


```

### Real data

```
python3 tests/end2end_tests/test_tabular_classification_diabetes.py -k test
python3 tests/end2end_tests/test_image_classification_mnist.py
python3 -m pytest tests/end2end_tests/test_tabular_regression_real_data.py -k test
python3 -m pytest tests/end2end_tests/test_timeseries_classification_real_data.py -k test
python3 -m pytest tests/end2end_tests/test_timeseries_regression_real_data.py -k test
python3 -m pytest tests/end2end_tests/test_timeseries_forecasting_real_data.py -k test

python3 -m pytest tests/end2end_tests/test_image_regression_real_data.py -k test
python3 -m pytest tests/end2end_tests/test_image_to_image_real_data.py -k test
python3 -m pytest tests/end2end_tests/test_image_to_image_real_data_extended.py -k test
python3 -m pytest tests/end2end_tests/test_image_segmentation_real_data.py -k test

python3 -m pytest tests/end2end_tests/test_text_regression_real_data.py -k test
python3 -m pytest tests/end2end_tests/test_text_classification_real_data.py -k test
python3 -m pytest tests/end2end_tests/test_text_to_text_real_data.py -k test
```