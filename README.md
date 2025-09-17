# Introduction to Quantization

As machine learning and artifiical intelligence have begun to take over our world, an interesting topic called quantization has arisen that has pretty interesting effects for the future of large models. Quantization is the process of mapping large (often continuous) set of values into a smaller finite set of values. You can think of it as rounding numbers to use fewer bits and save memory.

Example. Let's say I have all the numbers in the interval [0,1] and I want to use 8-bit precision. This means that I have 2<sup>8</sup> = 256 distinct values I can use to represent my numbers as.
So for a number like 0.5372, I'm  going to round it to a number in my distinct values.

**<p align="center"> 0.5372 &rarr; 0.54 </p>**

The error 0.54 - 0.5372 = 0.0028 is the difference we call quantization error since we lost precision. If we take this concept and apply it to machine learning, we know that models have millions to billions of parameters stored. Each of these parameters are stored as FP32 or 32-bit floating point numbers. This is a very precise computation but very expensive to compute and store which leads us to say that we can reduce this computation and storage expense by reducing the precision of these numbers using quantization (rounding).

# Quantization Methods

### Post-Training Quantization (PTQ)

In this method, you would first train a model in FP32 (32-bit floating point). Therefore, all the weights and biases would be of this precision. After training, we convert the weights/biases to the lower precision (Ex. INT8, INT4). This works well if the model isn't very sensitive to precision loss. It allows you to reduce the memory of storage and is less expensive to run inference (forward pass).

![Post-Training Quantization](quantization_ptq.png?refresh=1)

### Quantization-Aware Training (QAT)

The goal of this method is to create models that can still produce accurate results with quantized weights. During training of the model, we would simulate the quantization effects on the forward pass. This means we would quantize the weight when multiplying it to the input (Weight Quantization) and then quantize the output as well when sending it to the next layer (Activation Quantization). However, when we backpropagate, we still use the high precision so that the model is able to learn effectively. This method is particularly effective when you want to use less memory/compute (such as on mobile devices) but don't want to sacrifice precision.

#### Does this "simulated" quantization affect backpropagation?

If you have a background in machine learning, you'll easily see that we have a problem when doing backpropagation. When calculating forward pass, we are quantizing the weights and outputs through a function that looks like this

$$\hat{w} = round(\frac{w}{s_w}) * s_w$$
$$z = \hat{w}*x + \hat{b}$$
$$\hat{z} = round(\frac{z}{s_a}) * s_a$$

As we take the derivative with respect to the various parameters (gradient), we end up taking the derivative of the rounding function which is 0. This causes the gradient to disappear and we are unable to train our model correctly. Therefore, when we back-propagate, we will treat the rounding function as the identity function so that the derivative is 1 and it has no effect on the gradient being calculated.

### Why is Quantization Hard in LLMs?

1. Some weight matrices (attention...) can have big outlier values which causes the scaling from quantization to squash all the other values resulting in high precision loss
2. Large number of layers cause small quantization errors to accumulate and cause large drops in accuracy
3. Fitting large parameter models (70B+) would require high quantization (INT4, INT2) where quantization error becomes severe

# How to Use this Repo?

This repository contains clean cut python files for running/testing by yourself and also contains jupyter notebooks that walk you through each step of the process. For running the python files, you can use the following commands

| File Name | Sample Command |
|----------|--------|
| ptq_nn.py | python main.py --save_model=True |
| ptq_llm.py | python main.py --model_name=distilgpt2 --calib_samples=200 --seq_len=128 |
