import numpy as np
import jax
import math
from jax.scipy.special import logsumexp
import jax.numpy as jnp
from jax import grad, vmap, pmap, value_and_grad, jit
from jax import random
import time

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class MLP:
    """
    MLP using JAX and PyTorch
    """

    def __init__(self):
        self.params = []
        self.layer_width = [784, 512, 256, 10]
        self.key = jax.random.PRNGKey(1234)

        # Hyperparameters
        self.batch_size = 128
        self.lr = 0.001
        self.epochs = 20

    def init_weights(self):
        keys = jax.random.split(self.key, num=len(self.layer_width) - 1)

        for in_width, out_width, key in zip(
            self.layer_width[:-1], self.layer_width[1:], keys
        ):
            weight_key, bias_key = jax.random.split(key)
            bound = math.sqrt(6) / math.sqrt(in_width + out_width)
            self.params.append(
                [
                    jax.random.uniform(
                        weight_key,
                        shape=(out_width, in_width),
                        dtype=float,
                        minval=-bound,
                        maxval=bound,
                    ),
                    jax.random.uniform(bias_key, shape=out_width, dtype=float),
                ]
            )

    """def MLP_predict(self, x):
        activation = x
        for w, b in self.params[:-1]:
            activation = jax.nn.leaky_relu(jnp.dot(w, activation) + b)
        w_last, b_last = self.params[-1]
        logits = jnp.dot(w_last, activation) + b_last

        return logits - logsumexp(logits)"""

    def custom_transform(self, x):
        # Convert Torch tensor to np array
        return np.ravel(np.array(x, dtype=np.float32)) / 255.0

    def custom_collate_fn(self, batch):
        transposed_data = list(zip(*batch))
        labels = np.array(transposed_data[1])
        imgs = np.array(transposed_data[0])

        return imgs, labels

    def load(self):
        self.train_dataset = MNIST(
            "train_mnist", train=True, download=True, transform=self.custom_transform
        )
        self.test_dataset = MNIST(
            "test_mnist", train=False, download=True, transform=self.custom_transform
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            collate_fn=self.custom_collate_fn,
            drop_last=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=False,
            collate_fn=self.custom_collate_fn,
            drop_last=True,
        )

        self.batch_data = next(iter(self.train_loader))

        self.train_images = jnp.array(self.train_dataset.data).reshape(
            len(self.train_dataset), -1
        )

        self.train_labels = jnp.array(self.train_dataset.targets)

        self.test_images = jnp.array(self.test_dataset.data).reshape(
            len(self.test_dataset), -1
        )

        self.test_labels = jnp.array(self.test_dataset.targets)

        return (
            self.train_loader,
            self.test_loader,
            self.train_images,
            self.test_images,
            self.train_labels,
            self.train_labels,
        )

    def loss(self, params, imgs, gt_labels):
        # predictions = vmap(self.MLP_predict, in_axes=0)(imgs)  # (b,10)

        # return -jnp.mean(predictions * gt_labels)  # Cross-entropy
        # def loss(self, params, imgs, gt_labels):
        def predict_with_params(x):
            activation = x
            for w, b in params[:-1]:
                activation = jax.nn.leaky_relu(jnp.dot(activation, w.T) + b)
            w_last, b_last = params[-1]
            logits = jnp.dot(activation, w_last.T) + b_last
            return logits - logsumexp(logits)

        predictions = vmap(predict_with_params, in_axes=0)(imgs)
        return -jnp.mean(predictions * gt_labels)

    def update(self, imgs, gt_lbls):
        loss, grads = value_and_grad(self.loss)(self.params, imgs, gt_lbls)
        self.params = jax.tree.map(lambda p, g: p - self.lr * g, self.params, grads)
        return loss, self.params

    def accuracy(self, dataset_imgs, dataset_lbls):
        def predict_with_params(x):
            activation = x
            for w, b in self.params[:-1]:
                activation = jax.nn.leaky_relu(jnp.dot(activation, w.T) + b)
            w_last, b_last = self.params[-1]
            logits = jnp.dot(activation, w_last.T) + b_last
            return logits - logsumexp(logits)

        predictions = vmap(predict_with_params, in_axes=0)(dataset_imgs)

        pred_classes = jnp.argmax(predictions, axis=1)

        return jnp.mean(dataset_lbls == pred_classes)

    def training(self):
        # Loading
        self.load()

        for epoch in range(self.epochs):
            for cnt, (imgs, lbls) in enumerate(self.train_loader):
                gt_labels = jax.nn.one_hot(lbls, len(MNIST.classes))
                loss, self.params = self.update(imgs, gt_labels)

                if cnt % 50 == 0:
                    print(loss)
            print(
                f"Epoch = {epoch}, train_acc = {self.accuracy(self.train_images, self.train_labels)},"
                f"test_acc = {self.accuracy(self.test_images, self.test_labels)}"
            )

    def flow(self):
        self.init_weights()
        self.training()


if __name__ == "__main__":
    mlp = MLP()
    mlp_jit = MLP()
    mlp_pmap = MLP()

    print("NONE")
    print("----------")
    start = time.perf_counter()
    mlp.flow()
    end = time.perf_counter()
    elapsed = end - start

    print("JIT")
    print("----------")
    start_jit = time.perf_counter()
    jit(mlp_jit.flow)()
    end_jit = time.perf_counter()
    elapsed_jit = end_jit - start_jit
    print("----------")

    print(f"Time taken without jit: {elapsed:.6f} seconds")
    print(f"Time taken with jit: {elapsed_jit:.6f} seconds")
