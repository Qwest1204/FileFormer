
# Notus

a new solution to save space on your device. Imagine a bottomless handbag in which you can put anything you want, without restrictions. We've made it a reality!


## Authors

- Head of AI development: [@Qwest1204](https://github.com/Qwest1204)


## Tech Stack

**model:** PyTorch, Keras, numpy, BPETokenizer, Transformer

**train** RAY, Nvidia DGX, Selectel, Kubeflow

**data:** polars, hugging face


## Running Tests

To run the tests, run the following command

```bash
  python -m pytest --import-mode=append .
```


## Run Locally

request access to docker reg

```bash
docker pull ghcr.io/mralphafile/notus/notus-dev-cuda:latest
```


## Roadmap

- Train model on 10TB data

- Implement Reinforcement Learning to optimization on real data

- Implement full pipeline (request external info)


## Feedback

If you have any feedback, please reach out to us at notus@corp.com

