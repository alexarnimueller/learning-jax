import os
import urllib.request
from functools import partial
from urllib.error import HTTPError

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
import torch.utils.data as data
from flax.training import checkpoints, train_state
from IPython.display import set_matplotlib_formats
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import jax
from datasets import ReverseDataset
from jax import random
from models import TransformerPredictor
from optimizers import cosine_warmup_schedule
from utils import numpy_collate

matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()
set_matplotlib_formats("svg", "pdf")  # For export
plt.set_cmap("cividis")

main_rng = random.PRNGKey(42)

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "./data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./checkpoints/transformer"

print("Device:", jax.devices()[0])

# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial6/"
# Files to download
pretrained_files = ["ReverseTask.ckpt", "SetAnomalyTask.ckpt"]

# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if "/" in file_name:
        os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
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


class TrainerModule:

    def __init__(
        self,
        model_name,
        exmp_batch,
        max_iters,
        lr=1e-3,
        warmup=100,
        seed=42,
        **model_kwargs,
    ):
        """
        Inputs:
            model_name - Name of the model. Used for saving and checkpointing
            exmp_batch - Example batch to the model for initialization
            max_iters - Number of maximum iterations the model is trained for. Needed for the CosineWarmup scheduler
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            seed - Seed to use for model init
        """
        super().__init__()
        self.model_name = model_name
        self.max_iters = max_iters
        self.lr = lr
        self.warmup = warmup
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = TransformerPredictor(**model_kwargs)
        # Prepare logging
        self.log_dir = os.path.join(CHECKPOINT_PATH, self.model_name)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp_batch)

    def batch_to_input(self, exmp_batch):
        # Map batch to input data to the model
        # To be implemented in a task specific sub-class
        raise NotImplementedError

    def get_loss_function(self):
        # Return a function that calculates the loss for a batch
        # To be implemented in a task specific sub-class
        raise NotImplementedError

    def create_functions(self):
        # Create jitted train and eval functions
        calculate_loss = self.get_loss_function()

        # Training function
        def train_step(state, rng, batch):
            def loss_fn(params):
                calculate_loss(params, rng, batch, train=True)

            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, acc, rng = ret[0], *ret[1]
            state = state.apply_gradients(grads=grads)
            return state, rng, loss, acc

        self.train_step = jax.jit(train_step)

        # Evaluation function
        def eval_step(state, rng, batch):
            _, (acc, rng) = calculate_loss(state.params, rng, batch, train=False)
            return acc, rng

        self.eval_step = jax.jit(eval_step)

    def init_model(self, exmp_batch):
        # Initialize model
        self.rng = jax.random.PRNGKey(self.seed)
        self.rng, init_rng, dropout_init_rng = jax.random.split(self.rng, 3)
        exmp_input = self.batch_to_input(exmp_batch)
        params = self.model.init(
            {"params": init_rng, "dropout": dropout_init_rng}, exmp_input, train=True
        )["params"]
        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.lr,
            warmup_steps=self.warmup,
            decay_steps=self.max_iters,
            end_value=0.0,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(lr_schedule)
        )  # Clip gradients at norm 1
        # Initialize training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer
        )

    def train_model(self, train_loader, val_loader, num_epochs=500):
        # Train model for defined number of epochs
        best_acc = 0.0
        for epoch_idx in tqdm(range(1, num_epochs + 1)):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % 5 == 0:
                eval_acc = self.eval_model(val_loader)
                self.logger.add_scalar("val/accuracy", eval_acc, global_step=epoch_idx)
                if eval_acc >= best_acc:
                    best_acc = eval_acc
                    self.save_model(step=epoch_idx)
                self.logger.flush()

    def train_epoch(self, train_loader, epoch):
        # Train model for one epoch, and log avg loss and accuracy
        accs, losses = [], []
        for batch in tqdm(train_loader, desc="Training", leave=False):
            self.state, self.rng, loss, accuracy = self.train_step(
                self.state, self.rng, batch
            )
            losses.append(loss)
            accs.append(accuracy)
        avg_loss = np.stack(jax.device_get(losses)).mean()
        avg_acc = np.stack(jax.device_get(accs)).mean()
        self.logger.add_scalar("train/loss", avg_loss, global_step=epoch)
        self.logger.add_scalar("train/accuracy", avg_acc, global_step=epoch)

    def eval_model(self, data_loader):
        # Test model on all data points of a data loader and return avg accuracy
        correct_class, count = 0, 0
        for batch in data_loader:
            acc, self.rng = self.eval_step(self.state, self.rng, batch)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(
            ckpt_dir=self.log_dir, target=self.state.params, step=step
        )

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for the pretrained model
        if not pretrained:
            params = checkpoints.restore_checkpoint(
                ckpt_dir=self.log_dir, target=self.state.params
            )
        else:
            params = checkpoints.restore_checkpoint(
                ckpt_dir=os.path.join(CHECKPOINT_PATH, f"{self.model_name}.ckpt"),
                target=self.state.params,
            )
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=self.state.tx
        )

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this Transformer
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f"{self.model_name}.ckpt"))


class ReverseTrainer(TrainerModule):

    def batch_to_input(self, batch):
        inp_data, _ = batch
        inp_data = jax.nn.one_hot(inp_data, num_classes=self.model.num_classes)
        return inp_data

    def get_loss_function(self):
        # Function for calculating loss and accuracy for a batch
        def calculate_loss(params, rng, batch, train):
            inp_data, labels = batch
            inp_data = jax.nn.one_hot(inp_data, num_classes=self.model.num_classes)
            rng, dropout_apply_rng = random.split(rng)
            logits = self.model.apply(
                {"params": params},
                inp_data,
                train=train,
                rngs={"dropout": dropout_apply_rng},
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, labels
            ).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss, (acc, rng)

        return calculate_loss


def train_reverse(max_epochs=10, **model_args):
    num_train_iters = len(rev_train_loader) * max_epochs
    # Create a trainer module with specified hyperparameters
    trainer = ReverseTrainer(
        model_name="ReverseTask",
        exmp_batch=next(iter(rev_train_loader)),
        max_iters=num_train_iters,
        **model_args,
    )
    if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
        trainer.train_model(rev_train_loader, rev_val_loader, num_epochs=max_epochs)
        trainer.load_model()
    else:
        trainer.load_model(pretrained=True)
    val_acc = trainer.eval_model(rev_val_loader)
    test_acc = trainer.eval_model(rev_test_loader)
    # Bind parameters to model for easier inference
    trainer.model_bd = trainer.model.bind({"params": trainer.state.params})
    return trainer, {"val_acc": val_acc, "test_acc": test_acc}


lr_scheduler = cosine_warmup_schedule(base_lr=1.0, warmup=100, max_iters=2000)

# Test TransformerPredictor implementation
# Example features as input
main_rng, x_rng = random.split(main_rng)
x = random.normal(x_rng, (3, 16, 64))
# Create Transformer encoder
transpre = TransformerPredictor(
    num_layers=5,
    model_dim=128,
    num_classes=10,
    num_heads=4,
    dropout_prob=0.15,
    input_dropout_prob=0.05,
)
# Initialize parameters of transformer predictor with random key and inputs
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = transpre.init(
    {"params": init_rng, "dropout": dropout_init_rng}, x, train=True
)["params"]
# Apply transformer predictor with parameters on the inputs
# Since dropout is stochastic, we need to pass a rng to the forward
main_rng, dropout_apply_rng = random.split(main_rng)
# Instead of passing params and rngs every time to a function call, we can bind them to the module
binded_mod = transpre.bind({"params": params}, rngs={"dropout": dropout_apply_rng})
out = binded_mod(x, train=True)
print("Out", out.shape)
attn_maps = binded_mod.get_attention_maps(x, train=True)
print("Attention maps", len(attn_maps), attn_maps[0].shape)

del transpre, binded_mod, params

# train reverse
dataset = partial(ReverseDataset, 10, 16)
rev_train_loader = data.DataLoader(
    dataset(50000, np_rng=np.random.default_rng(42)),
    batch_size=128,
    shuffle=True,
    drop_last=True,
    collate_fn=numpy_collate,
)
rev_val_loader = data.DataLoader(
    dataset(1000, np_rng=np.random.default_rng(43)),
    batch_size=128,
    collate_fn=numpy_collate,
)
rev_test_loader = data.DataLoader(
    dataset(10000, np_rng=np.random.default_rng(44)),
    batch_size=128,
    collate_fn=numpy_collate,
)

reverse_trainer, reverse_result = train_reverse(
    model_dim=32,
    num_heads=1,
    num_classes=rev_train_loader.dataset.num_categories,
    num_layers=1,
    dropout_prob=0.0,
    lr=5e-4,
    warmup=50,
)

print(f"Val accuracy:  {(100.0 * reverse_result['val_acc']):4.2f}%")
print(f"Test accuracy: {(100.0 * reverse_result['test_acc']):4.2f}%")
