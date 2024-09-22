from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from rich.box import SIMPLE_HEAD
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (Progress,
                           BarColumn,
                           MofNCompleteColumn,
                           SpinnerColumn,
                           TaskProgressColumn,
                           TextColumn,
                           TimeElapsedColumn,
                           TimeRemainingColumn)
from rich.prompt import Confirm, Prompt
from rich.table import Table

from mnist import *

if TYPE_CHECKING:
    from typing import Sequence

TRAIN_DATASET = (
    "data/train-images-idx3-ubyte.gz",
    "data/train-labels-idx1-ubyte.gz"
)
T10K_DATASET = (
    "data/t10k-images-idx3-ubyte.gz",
    "data/t10k-labels-idx1-ubyte.gz"
)
DEFAULT_FILENAME = "model.npz"
HEIGHT = 28
WIDTH = 28

LAYERS = [16, 10]
BATCH_SIZE = 10
STEP = 1


def random_bw(sizes: "Sequence[int]") -> tuple[list[np.ndarray], list[np.ndarray]]:
    return ([np.random.random((1, j))*2-1 for j in sizes[1:]],
            [np.random.random((i, j))*2-1 for i, j in zip(sizes, sizes[1:])])


class Model(Perceptron):

    def __init__(self, input_size: "int"):
        sizes = [input_size, *LAYERS]
        super().__init__(*random_bw(sizes))

    def activate(self, x: "TFloatLike") -> "TFloatLike":
        return 1/(1+np.exp(-x))

    def activate_derivative(self, x: "TFloatLike") -> "TFloatLike":
        return np.exp(-x)/(1+np.exp(-x))**2

    def cost(self, y: "FloatArray", y0: "FloatArray") -> "Float":
        return np.sum((y-y0)**2) / y.size

    def cost_derivative(self, y: "FloatArray", y0: "FloatArray") -> "FloatArray":
        return 2*(y-y0)


def flatten_images(images: "np.ndarray") -> "np.ndarray":
    return images.reshape((*images.shape[:-2], 1, images.shape[-2]*images.shape[-1])) / 255


def transform_labels(labels: "np.ndarray") -> "np.ndarray":
    res = np.zeros((*labels.shape, 1, 10), dtype=np.float64)
    res[*(range(i) for i in labels.shape), 0, labels.flat] = 1.0
    return res


def recognize(model: 'Model', images: 'np.ndarray') -> "np.ndarray":
    output = model(images)
    index = np.amax(output, -1, keepdims=True) == output
    return np.broadcast_to(np.arange(10), index.shape)[index]


def main(console: 'Console'):
    console.clear()
    console.rule("Handwritten Digit Recognition")
    console.print("[bright_black]By osMrPigHead\n2024.09.17", justify="right")
    table = Table(pad_edge=False, show_header=False, show_edge=False)
    table.add_row("[red]1. Train & test new model")
    table.add_row("[yellow]2. Test existed model with MNIST")
    table.add_row("[cyan]3. Use existed model to recognize digits in \"data/images\" folder")
    console.print(table)
    selection = Prompt.ask("Select one to run",
                           console=console,
                           choices=["1", "2", "3"])
    console.print()
    match selection:
        case "1":
            model = _train(console)
            _test(console, model)
            _save(console, model)
        case "2":
            _test(console, _load(console))
        case "3":
            _recognize(console, _load(console))


def _train(console: 'Console') -> 'Model':
    with console.status("Preprocessing MNIST dataset..."):
        train_images = flatten_images(idx.load_from(TRAIN_DATASET[0], True))
        console.log("Training images loaded")
        train_labels = transform_labels(idx.load_from(TRAIN_DATASET[1], True))
        console.log("Training labels loaded")

        state = np.random.get_state()
        np.random.shuffle(train_images)
        np.random.set_state(state)
        np.random.shuffle(train_labels)

        batched_images = np.split(train_images,
                                  range(BATCH_SIZE, train_images.shape[0], BATCH_SIZE), 0)
        batched_labels = np.split(train_labels,
                                  range(BATCH_SIZE, train_labels.shape[0], BATCH_SIZE), 0)
        console.log("Training data preprocessed")

    model = Model(HEIGHT * WIDTH)

    table = Table(title="TRAINING", expand=True, pad_edge=False, show_header=False, show_edge=False)
    table.add_row(loss_c := Columns(["  Loss Rate", "FF.FF%"], width=12))
    table.add_row(progress := Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console
    ))
    with Live(Panel(table), console=console, refresh_per_second=20):
        task = progress.add_task("Training...", total=train_images.shape[0])
        for images, labels in zip(batched_images, batched_labels):
            dbl, dwl, loss = model(images, labels)
            loss_c.renderables[1] = "{: >5.2f}%".format(loss * 100)
            for b, w, db, dw in zip(model.layer_b, model.layer_w, dbl, dwl):
                b -= STEP*db
                w -= STEP*dw
            progress.update(task, advance=len(images))
        console.log("Model trained")

    return model


def _load(console: 'Console') -> 'Model':
    filename = Prompt.ask("Model filename", console=console, default=DEFAULT_FILENAME)
    try:
        return Model.load(filename)
    except OSError:
        console.print("[red]Not a valid model. Please enter an existed model file")
        _load(console)


def _test(console: 'Console', model: 'Model') -> 'None':
    with console.status("Loading MNIST dataset..."):
        t10k_images = flatten_images(idx.load_from(T10K_DATASET[0], True))
        console.log("Testing images loaded")
        t10k_labels = idx.load_from(T10K_DATASET[1], True)
        console.log("Testing labels loaded")

    with console.status("Testing..."):
        acc = np.sum(recognize(model, t10k_images) == t10k_labels) / t10k_labels.size
        console.log("Model tested. Accuracy: {: >5.2f}%".format(acc * 100))


def _recognize(console: 'Console', model: 'Model') -> 'None':
    imgs = []
    files = []
    with console.status("Preprocessing Images..."):
        for path in (Path("data") / "images").rglob("*.png"):
            if not path.is_file():
                continue
            imgs += [np.array(Image.open(path)
                              .resize((WIDTH, HEIGHT))
                              .convert("L"))]
            files += [path]
        if not imgs:
            console.log("[red]No image found")
            return
        console.log("Images loaded")
        imgs = flatten_images(np.stack(imgs))
    with console.status("Model running..."):
        res = recognize(model, imgs)
        console.log("Images recognized")
    table = Table("Image Filepath", "Recognized as", title="RECOGNITION RESULTS",
                  expand=True, box=SIMPLE_HEAD)
    for path, digit in sorted(zip(files, res)):
        table.add_row(str(path), str(digit))
    console.print(Panel(table))


def _save(console: 'Console', model: 'Model') -> 'None':
    if not Confirm.ask("[cyan]Save model to file?", console=console, default=True):
        return

    def __save():
        filename = Prompt.ask("Model filename", console=console, default=DEFAULT_FILENAME)
        try:
            model.save(filename)
        except OSError:
            console.print(f"[red]Error when saving to {filename}. Please enter a valid filename")
            __save()
    __save()


if __name__ == "__main__":
    main(Console())
