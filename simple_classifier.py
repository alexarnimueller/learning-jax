import numpy as np
import optax
import torch.utils.data as data
from flax.training import checkpoints, train_state
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from datasets import XORDataset
from models import SimpleClassifier
from plotting import visualize_classification, visualize_samples
from utils import numpy_collate

rng = jax.random.PRNGKey(42)  # PRNG state


def calculate_loss_acc(state, params, batch):
    data_input, labels = batch
    # Obtain the logits and predictions of the model for the input data
    logits = state.apply_fn(params, data_input).squeeze(axis=-1)
    pred_labels = (logits > 0).astype(jnp.float32)
    # Calculate the loss and accuracy
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (pred_labels == labels).mean()
    return loss, acc


@jax.jit  # Jit the function for efficiency
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(
        calculate_loss_acc,  # Function to calculate the loss
        argnums=1,  # Parameters are second argument of the function
        has_aux=True,  # Function has additional outputs, here accuracy
    )
    # Determine gradients for current model, parameters and batch
    (loss, acc), grads = grad_fn(state, state.params, batch)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss, acc


@jax.jit  # Jit the function for efficiency
def eval_step(state, batch):
    # Determine the accuracy
    _, acc = calculate_loss_acc(state, state.params, batch)
    return acc


def train_model(state, data_loader, num_epochs=100):
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for batch in data_loader:
            state, loss, acc = train_step(state, batch)
            # We could use the loss and accuracy for logging here, e.g. in TensorBoard
            # For simplicity, we skip this part here
    return state


def eval_model(state, data_loader):
    all_accs, batch_sizes = [], []
    for batch in data_loader:
        batch_acc = eval_step(state, batch)
        all_accs.append(batch_acc)
        batch_sizes.append(batch[0].shape[0])
    # Weighted average since some batches might be smaller
    acc = sum([a * b for a, b in zip(all_accs, batch_sizes)]) / sum(batch_sizes)
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")


model = SimpleClassifier(num_hidden=8, num_outputs=1)
# Printing the model shows its attributes
print(model)

# get rngs for data and model
rng, inp_rng, init_rng = jax.random.split(rng, 3)
inp = jax.random.normal(inp_rng, (8, 2))  # Batch size 8, input size 2

# Initialize the model and get parameters
params = model.init(init_rng, inp)
print(params)

# apply model to data
model.apply(params, inp)

# create dataset and visualize
dataset = XORDataset(size=200, seed=42)
print("Size of dataset:", len(dataset))
print("Data point 0:", dataset[0])
visualize_samples(dataset.data, dataset.label)

# instantiate a data loader on the dataset
data_loader = data.DataLoader(
    dataset, batch_size=8, shuffle=True, collate_fn=numpy_collate
)

# get a first batch of the data and print it
data_inputs, data_labels = next(iter(data_loader))
# The shape of the outputs are [batch_size, d_1,...,d_N] where d_1,...,d_N are the
# dimensions of the data point returned from the dataset class
print("Data inputs", data_inputs.shape, "\n", data_inputs)
print("Data labels", data_labels.shape, "\n", data_labels)

# create the optimizer
# Input to the optimizer are optimizer settings like learning rate
optimizer = optax.sgd(learning_rate=0.1)

# create a training state that bundels the parameters, the optimizer, and the forward step of the model:
model_state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=optimizer
)

# take a batch and calculate the loss on it
batch = next(iter(data_loader))
calculate_loss_acc(model_state, model_state.params, batch)

# training
train_dataset = XORDataset(size=2500, seed=42)
train_data_loader = data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, collate_fn=numpy_collate
)
trained_model_state = train_model(model_state, train_data_loader, num_epochs=100)

# save model state to checkpoint:
checkpoints.save_checkpoint(
    ckpt_dir="/home/arni/Documents/Code/jax/checkpoints/",  # Folder to save checkpoint in
    target=trained_model_state,  # What to save. To only save parameters, use model_state.params
    step=100,  # Training step or other metric to save best model on
    prefix="my_clf",  # Checkpoint file name prefix
    overwrite=True,  # Overwrite existing checkpoint files
)
# to load the model states again:
loaded_model_state = checkpoints.restore_checkpoint(
    ckpt_dir="./checkpoints/",  # Folder with the checkpoints
    target=model_state,  # (optional) matching object to rebuild state
    prefix="my_clf",  # Checkpoint file name prefix
)

# evaluation
test_dataset = XORDataset(size=500, seed=123)
# drop_last -> Don't drop the last batch although it is smaller than 128
test_data_loader = data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    drop_last=False,
    collate_fn=numpy_collate,
)

eval_model(trained_model_state, test_data_loader)

# to now use the model for predictions, bind it to the trained parameters
trained_model = model.bind(trained_model_state.params)

# now call it with new data
data_input, labels = next(iter(data_loader))
out = trained_model(data_input)  # No explicit parameter passing necessary anymore
out.shape

# visualize the classifier boundaries on the data
visualize_classification(trained_model, dataset.data, dataset.label)


# working with PyTrees (parameters)
parameters = jax.tree_leaves(model_state.params)
print(
    "We have parameters with the following shapes:",
    ", ".join([str(p.shape) for p in parameters]),
)
print("Overall parameter count:", sum([np.prod(p.shape) for p in parameters]))

# to create PyTrees, e.g. with all parameters, do as follows:
jax.tree_map(lambda p: p.shape, model_state.params)
