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
- overwrite basic function such as sum by a vriable name


##
