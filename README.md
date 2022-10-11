## L_inf PINN

This is the official implementation for [Is $L^2$ Physics-Informed Loss Always Suitable for Training Physics-Informed Neural Network?](https://arxiv.org/abs/2206.02016), which proposes a novel PINN training algorithm to minimize the $L^{\infty}$ loss in a similar spirit to adversarial training.

### Dependencies

The required packages are listed in `requirements.txt`, which can be installed by running `pip install -r requirements.txt`. 

### Getting started

To reproduce our result on 250-dimensional HJB Equation on a single GPU, run `python run.py`.

Multi-GPU training is also supported, e.g.,
```
python run.py hjb.gpu_cnt=2 \
      hjb.train.batch.domain_size=25 \
      hjb.train.batch.boundary_size=25
# shrink the batch size when there are mutiple GPUs
```

Training scripts for other experiments are provided in the `scripts` directory. For example, to train vanilla PINN on 100-dimensional HJB Equation, run `bash scripts/100-HJB-PINN.sh`.

### Contact

May you have any questions on our work or implementation, feel free to reach out to [`shandal@cs.cmu.edu`](shandal@cs.cmu.edu)!

### Citation

If you find this repository useful, please consider giving a star ⭐ and cite our paper.

```
@inproceedings{wang2022is,
      title={Is {$L^2$} Physics-Informed Loss Always Suitable for Training Physics-Informed Neural Network?}, 
      author={Chuwei Wang and Shanda Li and Di He and Liwei Wang},
      booktitle={Advances in Neural Information Processing Systems},
      year={2022},
}
```