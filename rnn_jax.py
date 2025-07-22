import numpy as np
import jax

from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)
import jax.numpy as jnp
from jax import grad, vmap, pmap, jit, value_and_grad
import optax

# from jax.experimental import optimizers as jax_opt
from jax.scipy.special import logsumexp
import torch.nn as nn
import torch

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Monitoring
from tqdm import tqdm
import time


# Data Preparation IMDB sentimens
class DataPrep:
    def __init__(self, test_path, train_path, batch_size: int):
        """Initialization function

        Args:
            str (train_path): path to store train dataset
            str (test_path): path to store test dataset
        """

        assert isinstance(train_path, str), "train_path must be a string"
        assert isinstance(test_path, str), "test_path must be a string"

        def custom_transform(x):
            """
            Transform for the Dataset
            """
            return np.ravel(np.array(x, dtype=np.float32)) / 255.0  # Normalize

        def custom_collate(batch):
            transposed_data = list(zip(*batch))
            labels = np.array(transposed_data[1])
            imgs = np.array(transposed_data[0])

            return imgs, labels

        self.train_dataset = CIFAR10(
            train_path, download=True, transform=custom_transform
        )
        self.test_dataset = CIFAR10(
            test_path, download=True, transform=custom_transform
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate,
            drop_last=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate,
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


class LSTMCell:
    """
    LSTM cell is made in the following way :
             ^h_t-1      Forget Gate
    |--------| -----> | mult | ---- | add |---------tanh--------------
                         |       ------|              |     h_t
        A               sig     sig   mult     sig--mult    |
                         |       |    tanh      |     |     |
    |--------| -----> | -----|--------------|---|     -----------x_t+1
    x_t-1            x_t            ^
                                Input Gate
    """

    def __init__(self, in_width: int, out_width: int):
        """_summary_ : Initialization of the weights

        Args:
            in_width (int): in dimension
            out_width (int): out dimension
            hidden
        """

        self.in_width = in_width
        self.out_width = out_width
        key = jax.random.PRNGKey(1234)
        # Split the key for different parameter initialization
        keys = jax.random.split(key, 8)

        # Xavier Init
        bound_input = jnp.sqrt(6.0 / (in_width + out_width))
        bound_hidden = jnp.sqrt(6.0 / (out_width + out_width))

        # Input-to-hidden weight 4 gates
        self.W_f = jax.random.uniform(
            keys[0], (in_width, out_width), minval=-bound_input, maxval=bound_input
        )
        self.W_i = jax.random.uniform(
            keys[1], (in_width, out_width), minval=-bound_input, maxval=bound_input
        )
        self.W_c = jax.random.uniform(
            keys[2], (in_width, out_width), minval=-bound_input, maxval=bound_input
        )
        self.W_o = jax.random.uniform(
            keys[3], (in_width, out_width), minval=-bound_input, maxval=bound_input
        )

        # Hiden-to-hiddein weights
        self.U_f = jax.random.uniform(
            keys[4], (out_width, out_width), minval=-bound_hidden, maxval=bound_hidden
        )
        self.U_i = jax.random.uniform(
            keys[5], (out_width, out_width), minval=-bound_hidden, maxval=bound_hidden
        )
        self.U_c = jax.random.uniform(
            keys[6], (out_width, out_width), minval=-bound_hidden, maxval=bound_hidden
        )
        self.U_o = jax.random.uniform(
            keys[7], (out_width, out_width), minval=-bound_hidden, maxval=bound_hidden
        )

        # Biases
        self.b_f = jnp.ones(out_width)
        self.b_i = jnp.zeros(out_width)
        self.b_c = jnp.zeros(out_width)
        self.b_o = jnp.zeros(out_width)

    def forward(self, x_t, h_prev, c_prev):
        """Single LSTM cell forward pass

            Args:
                x_t (array (numpy or jax)): input at time t, shape (batch_size, in_width)
                h_prev: previous hidden state, shape (batch_size, out_width)
            c_prev: previous cell state, shape (batch_size, out_width)

        Returns:
            h_t: new hidden state
            c_t: new cell state
        """
        # Forget cell
        f_t = jax.nn.sigmoid(
            jnp.dot(x_t, self.W_f) + jnp.dot(h_prev, self.U_f) + self.b_f
        )

        # Input cell
        i_t = jax.nn.sigmoid(
            jnp.dot(x_t, self.W_i) + jnp.dot(h_prev, self.U_i) + self.b_i
        )

        # Candidate values : new candidate values to add to cell state
        c_tilde = jax.nn.tanh(
            jnp.dot(x_t, self.W_c) + jnp.dot(h_prev, self.U_c) + self.b_c
        )

        # Uptade of the cell state
        c_t = f_t * c_prev + i_t * c_tilde

        # Output cell
        o_t = jax.nn.sigmoid(
            jnp.dot(x_t, self.W_o) + jnp.dot(h_prev, self.U_o) + self.b_o
        )

        # Update hidden state : output gate applied to cell state
        h_t = o_t * jnp.tanh(c_t)

        return h_t, c_t


class LSTM:
    def __init__(self, in_width, hidden_width, num_layers=1, num_classes=10):
        """Back bone of LSTM architecture using all the cells where the foward pass is made

        Args:
            in_width (_type_): in_width dimension
            hidden_width (_type_): hidden state dimension
            num_layers (int, optional): number of layer in the architecture. Defaults to 1.
            num_classes : Number of classes in the CIFAR-10 Dataset
        """
        # Dimension initialization
        self.hidden_width = hidden_width
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Create an LSTM cell for each layer
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, num_layers + 1)

        # LSTM cells
        self.lstm_cells = []
        for i in range(num_layers):
            input_dim = in_width if i == 0 else hidden_width
            cell = LSTMCell(input_dim, hidden_width)
            self.lstm_cells.append(cell)

        # Classification head
        bound = jnp.sqrt(6.0 / (hidden_width + num_classes))
        self.W_out = jax.random.uniform(
            keys[-1], (hidden_width, num_classes), minval=-bound, maxval=bound
        )

        self.b_out = jnp.zeros(num_classes)

    def init_hidden(self, batch_size):
        """Initialize hidden and cell state"""
        h_states = [
            jnp.zeros((batch_size, self.hidden_width)) for _ in range(self.num_layers)
        ]
        c_states = [
            jnp.zeros((batch_size, self.hidden_width)) for _ in range(self.num_layers)
        ]
        return h_states, c_states

    def forward(self, sequence):
        """Forward pass throught the LSTM

        Args:
            sequence : Input sequence

        Returns:
            softmax of logits
        """
        batch_size, seq_lenght, _ = sequence.shape
        h_states, c_states = self.init_hidden(batch_size)

        # Process sequence through all timesteps
        for t in range(seq_lenght):
            x_t = sequence[:, t, :]  # Current input

            # Pass through each LSTM layer
            for layer_idx in range(self.num_layers):
                # Use of vmap to vectorize
                h_states[layer_idx], c_states[layer_idx] = self.lstm_cells[
                    layer_idx
                ].forward(x_t, h_states[layer_idx], c_states[layer_idx])
                x_t = h_states[layer_idx]

        # Use the final hidden state for classification
        final_hidden = h_states[-1]
        logits = jnp.dot(final_hidden, self.W_out) + self.b_out
        return logits - logsumexp(logits, axis=1, keepdims=True)

    def reshape_cifar_for_rnn(self, images):
        """
        Reshape CIFAR-10 images for RNN processing

        Args:
            images: shape (batch_size, 3072) - flattened CIFAR images

        Returns:
            sequence: shape (batch_size, 32, 96) - treat as 32 timesteps of 96 features
        """
        batch_size = images.shape[0]
        # Reshape 3072 -> 32 x 96 (32 timesteps, 96 features per timestep)
        return images.reshape(batch_size, 32, 96)


class Training:
    def __init__(self, model, learning_rate: float, epochs: int):
        """
        Training class for the LSTM

        Args:
            model (_type_): LSTM model
            learning_rate (float): Learning rate for gradient backprop
            epochs (int): Number of epochs in the training
        """

        self.model = model
        self.lr = learning_rate
        self.epochs = epochs

        # Optimizer Adam
        self.optimizer = optax.adam(learning_rate)

        # Get all model parameters
        self.params = self.get_model_params()

        # Initialize optimizer status
        self.opt_state = self.optimizer.init(self.params)

    def get_model_params(self):
        """Extract all the params of the model"""

        params = {}

        # LSTM layer parameters
        for i, cell in enumerate(self.model.lstm_cells):
            params[f"layer_{i}"] = {
                "W_f": cell.W_f,
                "W_i": cell.W_i,
                "W_c": cell.W_c,
                "W_o": cell.W_o,
                "U_f": cell.U_f,
                "U_i": cell.U_i,
                "U_c": cell.U_c,
                "U_o": cell.U_o,
                "b_f": cell.b_f,
                "b_i": cell.b_i,
                "b_c": cell.b_c,
                "b_o": cell.b_o,
            }

        # Output layer parameters
        params["output"] = {"W_out": self.model.W_out, "b_out": self.model.b_out}

        return params

    def update_params(self, params):
        """Update model params"""

        # Update LSTM
        for i, cell in enumerate(self.model.lstm_cells):
            layer_params = params[f"layer_{i}"]
            cell.W_f, cell.W_i, cell.W_c, cell.W_o = (
                layer_params["W_f"],
                layer_params["W_i"],
                layer_params["W_c"],
                layer_params["W_o"],
            )
            cell.U_f, cell.U_i, cell.U_c, cell.U_o = (
                layer_params["U_f"],
                layer_params["U_i"],
                layer_params["U_c"],
                layer_params["U_o"],
            )
            cell.b_f, cell.b_i, cell.b_c, cell.b_o = (
                layer_params["b_f"],
                layer_params["b_i"],
                layer_params["b_c"],
                layer_params["b_o"],
            )

        # Update output layer
        self.model.W_out = params["output"]["W_out"]
        self.model.b_out = params["output"]["b_out"]

    def loss_fn(self, params, images, labels):
        """Loss function that takes parameters explicitly

        Args:
            params (_type_): _description_
            images (_type_): _description_
            labels (_type_): _description_
        """

        old_params = self.get_model_params()
        self.update_params(params)

        # Forward pass
        sequences = self.model.reshape_cifar_for_rnn(images)
        softmax_logits = self.model.forward(sequences)

        # One hot encoding on the labels
        one_hot_labels = jax.nn.one_hot(labels, self.model.num_classes)

        loss = -jnp.mean(jnp.sum(one_hot_labels * softmax_logits, axis=1))

        self.update_params(old_params)

        return loss

    def train_step(self, params, opt_state, images, labels):
        """Single training step with JIT compilation

        Args:
            params (_type_): _description_
            opt_state (_type_): _description_
            images (_type_): _description_
            labels (_type_): _description_
        """
        loss, grads = value_and_grad(self.loss_fn)(params, images, labels)

        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)

        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss

    def accuracy(self, images, labels):
        """Compute accuracy

        Args:
            images (_type_): _description_
            labels (_type_): _description_
        """
        sequences = self.model.reshape_cifar_for_rnn(images)
        logits = self.model.forward(sequences)
        predictions = jnp.argmax(logits, axis=1)
        return jnp.mean(predictions == labels)

    def train(self, data_prep):
        """Main loop train

        Args:
            data_prep (_type_): _description_
        """
        # Move full datasets to GPU for accuracy computation
        train_images_gpu = jax.device_put(data_prep.train_images[:1000] / 255.0)
        train_labels_gpu = jax.device_put(data_prep.train_labels[:1000])
        test_images_gpu = jax.device_put(data_prep.test_images[:1000] / 255.0)
        test_labels_gpu = jax.device_put(data_prep.test_labels[:1000])

        print("Starting training")
        for epoch in range(self.epochs):
            print(f"Epoch = {epoch}")
            epoch_loss = 0.0
            num_batches = 0

            start_time = time.time()

            for batch_imgs, batch_labels in data_prep.train_loader:
                # Convert batch imgs & labels JAX array in the DataPrep class
                imgs = jax.device_put(jnp.array(batch_imgs))
                labels = jax.device_put(jnp.array(batch_labels))
                # Single training step

                self.params, self.opt_state, loss = self.train_step(
                    self.params, self.opt_state, imgs, labels
                )

                self.update_params(self.params)

                epoch_loss += loss
                num_batches += 1

            # Compute accuracies

            train_acc = self.accuracy(train_images_gpu[:1000], train_labels_gpu[:1000])
            test_acc = self.accuracy(test_images_gpu[:1000], test_labels_gpu[:1000])

            epoch_time = time.time() - start_time
            avg_loss = epoch_loss / num_batches

            print(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Loss: {avg_loss:.4f} - "
                f"Train Acc: {train_acc:.4f} - "
                f"Test Acc: {test_acc:.4f} - "
                f"Time: {epoch_time:.2f}s"
            )


if __name__ == "__main__":
    if jax.default_backend() == "gpu":
        print("✅ JAX is using GPU!")
        print(f"GPU memory: {jax.devices()[0].memory_stats()}")
    else:
        print("❌ JAX is using CPU - check GPU installation")
    # Initialize data and model
    data_prep = DataPrep("./data", "./data", batch_size=32)
    lstm_model = LSTM(in_width=96, hidden_width=128, num_layers=2, num_classes=10)

    # Initialize trainer
    trainer = Training(lstm_model, learning_rate=0.01, epochs=10)

    # Start training
    trainer.train(data_prep)
