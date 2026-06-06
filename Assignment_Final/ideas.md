## Dataset
- Work with MedMnist https://github.com/MedMNIST/MedMNIST
- Benchmarking was done here: https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus
- Description of datasets https://arxiv.org/pdf/2110.14795
- Use PathMnist as RGB (colorectal cancer histology slides) and TissueMnist as grayscale (Confocal single channel images of DAPI fluorescence)
- Prepare data manually so not obvious it is the med mnist - extract data into torch tensors
    - rename to histology_data.pt and dapi_data.pt

## Assignment structure
- code initially working for dapi, wanted to extend to histology - all broke down
- fix number of classes to 8 - will break for histology
- fix number of channels to 3 - will break for dapi
- no longer have the config file
- dapi works without normalization but histology completely breaks - needs to be introduced in get_loaders
- request confusion matrix to spot the missing class problem
- overwrite basic function such as sum by a vriable name?

Suggested Assignment Prompt Addendum
📊 Metric Requirement: Beyond Global Accuracy
"In medical imaging, overall classification accuracy is an incomplete metric. A model can achieve a high overall score while completely failing on specific rare pathologies or suffering from data pipeline mismatches.

Therefore, you are strictly required to evaluate your final model by calculating and displaying per-class metrics (Precision, Recall, and F1-score for each individual class index) using validation data.

Operational Directive: If your per-class breakdown reveals any unaligned categories, categories with zero support, or structural anomalies that do not match the real-world dataset footprint, you must trace the root configuration mismatch and rectify it in your codebase."


## List of corrupts
1. Missing config file
2. Missing normalization in get_loaders - will work for dapi, but histology very bad performance
3. in_channels=3 and num_classes=9 as default in class ResNet
4. drop in_channels and num_classes args from ResNet instatiation in main
5. sum as variable in Trainer
6. droped labels = labels.squeeze().long() from Trainer
7. drop zero_out gradients
