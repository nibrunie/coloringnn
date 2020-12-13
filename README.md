# coloringnn
CNN to color black and white pictures (Coloring convolutional Neural Network)

=== Install
see `requirements.txt` for a dependency list

To install python dependencies you can use `pip`
```
pip install -r requirements.txt
```

=== Training

You can use the following command to prepare a dataset from a directory containing color images:
```
python3 prepare_dataset.py --dataset-size 1000 --augment --output <output directory>  <input directory>
```

Then you can execute training on prepared dataset:
```
python3 coloring_network.py --model-path <path to save/restore model> --dataset <dataset directory> --dataset-size 10000 --train 100 --shuffle-dataset
```

=== Inference
```
python3 coloring_network.py --model-path <path to .model to use> --eval-on-img <image under test>
```
