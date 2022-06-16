# Continuous Wasserstein-1 Benchmark
This is the official `Python` implementation of the benchmark paper **Kantorovich Strikes Back! Wasserstein GANs are not Optimal Transport?** by [Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en), [Alexander Kolesov](https://scholar.google.com/citations?user=vX2pmScAAAAJ&hl=ru&oi=ao)  and [Evgeny Burnaev](https://scholar.google.ru/citations?user=pCRdcOwAAAAJ&hl=ru).

The repository contains a set of continuous benchmark distributions for testing optimal transport solvers with the distance cost (the Wasserstein-1 distance), the code for optimal transport solvers and their evaluation.

 
## Pre-requisites
The implementation is GPU-based. Single GPU (~GTX 1080 ti) is enough to run each particular experiment. Tested with

`torch==1.10.2 `

The code might not run as intended in newer `torch` versions.

## Related repositories
- [Repository](https://github.com/iamalexkorotin/Wasserstein2Benchmark) for [Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark](https://openreview.net/forum?id=CI0T_3l-n1) paper.

## Loading Benchmark Pairs
```python
import sys
sys.path.append("..")
from src.map_benchmark import MixToOneBenchmark, Celeba64Benchmark

# Make high-dimensional benchmark for dimensions 16(2, 4, ..., 128)
#for number of funnels 64(4, 16, 64, 256)
benchmark =  MixToOneBenchmark(16,64)

# OR create 1-Lipschitz net for  images benchmark pair 
# for number of funnels  16 (1,16), degree 10 (10, 100) correspondingly
# benchmark = Celeba64Benchmark( 16, 10)


# Sample 32 random points from the benchmark distributions
X = benchmark.input_sampler.sample(32)
Y = benchmark.output_sampler.sample(32)

# Sample 32 random points from the OT plan
# X, Y = benchmark.input_sampler.sample(32 ,flag_plan = True)
```

## Repository structure
- `notebooks/` - jupyter notebooks with preview of benchmark pairs and the evaluation of OT solvers;
- `src/` - auxiliary source code;
- `metrics/` - results of the evaluation (cosine, L2, W1);
- `benchmarks/` - `.pt` checkpoints for continuous benchmark pairs.

## Evaluation of Existing WGAN OT Solvers
We provide all the code to evaluate existing dual WGAN OT solvers on our benchmark pairs. The qualitative results are shown below. For quantitative results, see the paper or `metrics/`.

### High-Dimensional Benchmarks
- `notebooks/get_benchmark_nd.ipynb` -- previewing benchmark pairs;
- `notebooks/test_nd.ipynb` -- testing dual WGAN OT solvers.

The pipeline of the proposed method for constructing benchmark pairs.

<p align="center"><img src="pics/nd/plot_teaser.png" width="700" /></p>

The surfaces of the potential learned by the OT solvers in 2-dimensional experiment with 4 funnels.

<p align="center"><img src="pics/nd/plot_surfaces.png" width="700" /></p>

### Images Benchmark Pairs 
- `notebooks/get_benchmark_celeba.ipynb` -- previewing benchmark pairs;
- `notebooks/test_celeba.ipynb` -- testing dual WGAN OT solvers.

<p  align="center">
  <img src= "pics/celeba/celeba_faces_1.png" width="400" />
  <img src="pics/celeba/celeba_faces_16.png" width="400" /> 
</p>

 
<p  align="center">
  <img src= "pics/celeba/pca_celeba.png" width="300" />
  <img src="pics/celeba/pca_celeba_16.png" width="200" /> 
  <img src="pics/celeba/pca_celeba_1.png" width="300" />  
</p>
 
## Credits
- [CelebA page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) with faces dataset;
- [UNet architecture](https://github.com/milesial/Pytorch-UNet) for the mover;
- [ResNet architectures](https://github.com/harryliew/WGAN-QC) for the generator;
- [DC-GAN architecture](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) for the discriminator;