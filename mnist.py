import json
import os
import pickle
import urllib.request
from urllib.error import HTTPError

import flax.linen as nn
import matplotlib.pyplot as plt
import optax
import seaborn as sns
import torch
import torch.utils.data as data
from flax.training import train_state
from torchvision.datasets import FashionMNIST
from tqdm.auto import tqdm

import jax
from activations import act_fn_by_name
from jax import random
from models import BaseNetwork
from optimizers import adam
from utils import image_to_numpy, numpy_collate, xavier_init

CHECKPOINT_PATH = "./checkpoints"
DATASET_PATH = "./data"

print("Device:", jax.devices()[0])

# download pretrained models
# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial4/"
pretrained_files = [
    "FashionMNIST_SGD.config",
    "FashionMNIST_SGD_results.json",
    "FashionMNIST_SGD.tar",
    "FashionMNIST_SGDMom.config",
    "FashionMNIST_SGDMom_results.json",
    "FashionMNIST_SGDMom.tar",
    "FashionMNIST_Adam.config",
    "FashionMNIST_Adam_results.json",
    "FashionMNIST_Adam.tar",
]

# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print(
                "Something went wrong. Please try to download the file from the GDrive folder, or contact the author "
                + "with the full output including the following error:\n",
                e,
            )


# Loading the training dataset. We need to split it into a training and validation part
train_dataset = FashionMNIST(
    root=DATASET_PATH, train=True, transform=image_to_numpy, download=True
)
train_set, val_set = torch.utils.data.random_split(
    train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42)
)

# Loading the test set
test_set = FashionMNIST(
    root=DATASET_PATH, train=False, transform=image_to_numpy, download=True
)

# We define a set of data loaders that we can use for various purposes later.
# Note that for actually training a model, we will use different data loaders
# with a lower batch size.
train_loader = data.DataLoader(
    train_set, batch_size=1024, shuffle=False, drop_last=False, collate_fn=numpy_collate
)
val_loader = data.DataLoader(
    val_set, batch_size=1024, shuffle=False, drop_last=False, collate_fn=numpy_collate
)
test_loader = data.DataLoader(
    test_set, batch_size=1024, shuffle=False, drop_last=False, collate_fn=numpy_collate
)

# some example images
small_loader = data.DataLoader(
    train_set, batch_size=1024, shuffle=False, collate_fn=numpy_collate
)
exmp_imgs, exmp_labels = next(iter(small_loader))


def init_simple_model(kernel_init, act_fn=act_fn_by_name["identity"], samples=None):
    model = BaseNetwork(act_fn=act_fn, kernel_init=kernel_init)
    params = model.init(random.PRNGKey(42), samples)
    return model, params


def _get_config_file(model_path, model_name):
    # Name of the file for storing hyperparameter details
    return os.path.join(model_path, model_name + ".config")


def _get_model_file(model_path, model_name):
    # Name of the file for storing network parameters
    return os.path.join(model_path, model_name + ".tar")


def _get_result_file(model_path, model_name):
    return os.path.join(model_path, model_name + "_results.json")


def load_model(model_path, model_name, state=None):
    """
    Loads a saved model from disk.

    Inputs:
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
        state - (Optional) If given, the parameters are loaded into this training state. Otherwise,
                a new one is created alongside a network architecture.
    """
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(
        model_path, model_name
    )
    assert os.path.isfile(config_file), (
        f'Could not find the config file "{config_file}". '
        + "Are you sure this is the correct path and you have your model config stored here?"
    )
    assert os.path.isfile(model_file), (
        f'Could not find the model file "{model_file}"'
        + "Are you sure this is the correct path and you have your model stored here?"
    )
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    if state is None:
        net = BaseNetwork(act_fn=nn.relu, **config_dict)
        state = train_state.TrainState(
            step=0, params=None, apply_fn=net.apply, tx=None, opt_state=None
        )
    else:
        net = None
    # You can also use flax's checkpoint package. To show an alternative,
    # you can instead load the parameters simply from a pickle file.
    with open(model_file, "rb") as f:
        params = pickle.load(f)
    state = state.replace(params=params)
    return state, net


def save_model(model, params, model_path, model_name):
    """
    Given a model, we save the parameters and hyperparameters.

    Inputs:
        model - Network object without parameters
        params - Parameters to save of the model
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
    """
    config_dict = {"hidden_sizes": model.hidden_sizes, "num_classes": model.num_classes}
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(
        model_path, model_name
    )
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    # You can also use flax's checkpoint package. To show an alternative,
    # you can instead save the parameters simply in a pickle file.
    with open(model_file, "wb") as f:
        pickle.dump(params, f)


def calculate_loss(params, apply_fn, batch):
    imgs, labels = batch
    logits = apply_fn(params, imgs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    acc = (labels == logits.argmax(axis=-1)).mean()
    return loss, acc


@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
    (_, acc), grads = grad_fn(state.params, state.apply_fn, batch)
    state = state.apply_gradients(grads=grads)
    return state, acc


@jax.jit
def eval_step(state, batch):
    _, acc = calculate_loss(state.params, state.apply_fn, batch)
    return acc


def train_model(
    net, params, optimizer, model_name, max_epochs=50, batch_size=256, overwrite=False
):
    """
    Train a model on the training set of FashionMNIST

    Inputs:
        net - Object of BaseNetwork
        params - The parameters to use as initialization
        optimizer - Optimizer to use
        model_name - (str) Name of the model, used for creating the checkpoint names
        max_epochs - Number of epochs we want to (maximally) train for
        batch_size - Size of batches used in training
        overwrite - Determines how to handle the case when there already exists a checkpoint.
                    If True, it will be overwritten. Otherwise, we skip training.
    """
    file_exists = os.path.isfile(_get_model_file(CHECKPOINT_PATH, model_name))
    if file_exists and not overwrite:
        print("Model file already exists. Skipping training...")
        state = None
        with open(_get_result_file(CHECKPOINT_PATH, model_name), "r") as f:
            results = json.load(f)
    else:
        if file_exists:
            print("Model file exists, but will be overwritten...")

        # Initializing training state
        results = None
        state = train_state.TrainState.create(
            apply_fn=net.apply, params=params, tx=optimizer
        )

        # Defining data loader
        train_loader_local = data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=numpy_collate,
            generator=torch.Generator().manual_seed(42),
        )

        train_scores = []
        val_scores = []
        best_val_epoch = -1
        for epoch in range(max_epochs):
            ############
            # Training #
            ############
            train_acc = 0.0
            for batch in tqdm(train_loader_local, desc=f"Epoch {epoch+1}", leave=False):
                state, acc = train_step(state, batch)
                train_acc += acc
            train_acc /= len(train_loader_local)
            train_scores.append(train_acc.item())

            ##############
            # Validation #
            ##############
            val_acc = test_model(state, val_loader)
            val_scores.append(val_acc)
            print(
                f"[Epoch {epoch+1:2d}] Training accuracy: {train_acc:05.2%}, Validation accuracy: {val_acc:4.2%}"
            )

            if len(val_scores) == 1 or val_acc > val_scores[best_val_epoch]:
                print("\t   (New best performance, saving model...)")
                save_model(net, state.params, CHECKPOINT_PATH, model_name)
                best_val_epoch = epoch

    state, _ = load_model(CHECKPOINT_PATH, model_name, state=state)
    if results is None:
        test_acc = test_model(state, test_loader)
        results = {
            "test_acc": test_acc,
            "val_scores": val_scores,
            "train_scores": train_scores,
        }
        with open(_get_result_file(CHECKPOINT_PATH, model_name), "w") as f:
            json.dump(results, f)

    # Plot a curve of the validation accuracy
    sns.set()
    plt.plot(
        [i for i in range(1, len(results["train_scores"]) + 1)],
        results["train_scores"],
        label="Train",
    )
    plt.plot(
        [i for i in range(1, len(results["val_scores"]) + 1)],
        results["val_scores"],
        label="Val",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Validation accuracy")
    plt.ylim(min(results["val_scores"]), max(results["train_scores"]) * 1.01)
    plt.title(f"Validation performance of {model_name}")
    plt.legend()
    plt.show()
    plt.close()

    print((f" Test accuracy: {results['test_acc']:4.2%} ").center(50, "=") + "\n")
    return state


def test_model(state, data_loader):
    """
    Test a model on a specified dataset.

    Inputs:
        state - Training state including parameters and model apply function.
        data_loader - DataLoader object of the dataset to test on (validation or test)
    """
    true_preds, count = 0.0, 0
    for batch in data_loader:
        acc = eval_step(state, batch)
        batch_size = batch[0].shape[0]
        true_preds += acc * batch_size
        count += batch_size
    test_acc = true_preds / count
    return test_acc.item()


# initialization
model, params = init_simple_model(xavier_init, act_fn=nn.relu, samples=exmp_imgs)
_ = train_model(
    model, params, adam(lr=1e-3), "FashionMNIST_Adam", max_epochs=40, batch_size=256
)
