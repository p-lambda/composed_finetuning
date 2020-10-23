# Simplifying Models with Unlabeled Output Data

This repo provides the code accompanying the paper "Simplifying Models with Unlabeled Output Data", which proposes the predict-and-denoise framework for prediction problems with high-dimensional outputs. In this framework, we first use "unlabeled" output data, i.e. outputs without corresponding inputs that are often freely available, the learn a denoiser from perturbed, noisy outputs to valid outputs. We then train a predictor composed with this (fixed) denoiser, leveraging the output structure learned by the denoiser to improve sample complexity.
Predict-and-denoise can leverage unlabeled output data to improve models especially in situations when there is only a small amount of labeled data available.

We provide experiments on both images and text:
- conditional image generation
- pseudocode to code (under construction)

If this repository was useful for you, please cite:
```
@article{xie2020simplifying, 
title={Simplifying models with unlabeled output data}, 
author={Xie, Sang Michael and Ma, Tengyu and Liang, Percy},
journal={arXiv preprint arXiv:2006.16205}, 
year={2020} 
}
```

# Conditional image generation experiments

The predict-and-denoise framework can leverage unlabeled output data in conditional image generation, especially when there is only a small amount of labeled data available.  We demonstrate this in a font image generation experiment, where the input is the font type and character identity and the output is an image of the font.
The commands to run the experiments are in `imggen/runner.sh`.
Our experiments train a simple feedforward network for conditional image generation. We first train a U-Net to denoise unlabeled font images perturbed with a Gaussian blur filter. We then train the feedforward network composed with the (fixed) denoiser. With appropriate regularization, the feedforward network (base predictor) learns a much simpler function, without the responsibility of outputting an image with sharply defined lines. The complexity is offloaded to the denoiser, which has the advantage of having much more training data.

Here is an example of the differences between a Direct predictor trained directly to predict images, and the Composed predictor which factorizes the problem with the predict-and-denoise framework. The Composed predictor as a whole outputs clearer images than the Direct predictor, while the first part of the composed predictor (the base predictor) outputs gray, blurry images that can be fit with a simple, low-norm solution.

![font generation](imggen/font-generation.png)

