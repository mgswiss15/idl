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
- request thorough evaluation over test data

## List of corrupts in data
1. Missing normalization
2. Validation set data leakackage

## List of corrupts in Trainer
1. sum as variable in Trainer
2. droped labels = labels.squeeze().long() from Trainer
3. drop zero_out gradients



## List of corrupts in models
1. VGGBlock: padding always 1 even for 1x1 convolution. removed `padding = 0 if is_config_c_tail else 1`
2. hardcoded in_channels=3 and num_classes=11 in AlexNet
3. AlexNet classifier wrong in_dim: 2048 istead of 3072
4. nn.Softmax(dim=1) in VGG net
5. drop_rate=0.99 in train.py to stop training AlexNet and VGG net
6. drop `current_in_channels = out_channels` from VGG block
7. drop `activation_str = kwargs.get("activation", "ReLU")` from ResNet - kept the printing to make it easier ;)

