# torch-model-sharding

Sharding **weights.pt** from torch.save to multiple **pytorch_model_0000x-of-0000x.bin** files for fast upload.

## install

```bash
pip install torch-model-sharding==1.0.0
```

## how to use command-line

```bash
torch-model-sharding --source_file=/path/to/torch/pt/file --sharding_factor=3 --target_dir=/path/to/target/directory
```

## dev

```bash
## build
python3 setup.py sdist

## publish
twine upload dist/*

```
