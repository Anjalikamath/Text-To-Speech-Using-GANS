# Text-to-Speech Model employing Content-Style Disentanglement using GANS

This is the code for the natual language generating Text-to-Speech model built upon Generative Adversarial Networks. It employs content-stle disentanglement and synthesizes high fidelity audio with the correct verbal content and the desired auditory style and tone. 

## Preprocess
```bash
python preprocess/make_dataset_vctk.py vctk.h5
python preprocess/make_single_samples.py vctk.h5 index.json
```

## Training
```bash
python main.py
```
