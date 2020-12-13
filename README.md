# coloringnn
CNN to color black and white pictures (Coloring convolutional Neural Network)

=== Install
see `requirements.txt` for a dependency list

To install python dependencies you can use `pip`
```
pip install -r requirements.txt
```

=== Training
```
python3 coloring_network.py --model-path <path to save/restore model> --dataset <dataset directory> --dataset-size 10000 --train 100 --shuffle-dataset
```

=== Inference
```
python3 coloring_network.py --model-path <path to .model to use> --eval-on-img <image under test>
```
