# TODO: Add Metrics Printing After Each Epoch

## Completed Tasks
- [x] Create custom `PrintMetricsCallback` class to print training accuracy, training loss, testing accuracy, testing loss, validation accuracy, and validation loss after each epoch.
- [x] Instantiate the callback with `test_gen` and `train_gen`.
- [x] Add the callback to `callbacks_phase1`, `callbacks_phase2`, and `callbacks_phase3` to ensure it runs during all training phases.

## Followup Steps
- [ ] Run the `train_efficientnet_proper.py` script to verify that the metrics are printed correctly after each epoch in all phases.
- [ ] Check the output to ensure training accuracy, training loss, testing accuracy, testing loss, validation accuracy, and validation loss are displayed as expected.
