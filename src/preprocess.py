import random
import re
from typing import Callable, List, Tuple

from datasets import load_dataset as hf_load_dataset


def extract_last_int(text: str) -> int | None:
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def extract_last_ab(text: str) -> str | None:
    matches = re.findall(r"\b(A|B)\b", text)
    return matches[-1] if matches else None


def get_normalization_fn(name: str) -> Callable[[str], int | str | None]:
    if name == "extract_last_int":
        return extract_last_int
    if name == "extract_last_ab":
        return extract_last_ab
    raise ValueError(f"Unknown normalization: {name}")


def make_prompt(question: str) -> str:
    return f"Q: {question}\nA:"


OPS = ["+", "-", "*"]


def gen_arith(steps: int = 3, bound: int = 9) -> Tuple[str, int]:
    x = random.randint(1, bound)
    expr = str(x)
    desc = [f"Start with {x}."]
    for _ in range(steps):
        op = random.choice(OPS)
        a = random.randint(1, bound)
        if op == "+":
            desc.append(f"Add {a}.")
        elif op == "-":
            desc.append(f"Subtract {a}.")
        else:
            desc.append(f"Multiply by {a}.")
        expr = f"({expr}{op}{a})"
    y = int(eval(expr))
    q = " ".join(desc) + " What is the result?"
    return q, y


def gen_coinflip(n: int = 4) -> Tuple[str, str]:
    flips = [random.choice(["flip", "stay"]) for _ in range(n)]
    state = "A"
    desc = ["A coin starts on side A."]
    for f in flips:
        if f == "flip":
            state = "B" if state == "A" else "A"
            desc.append("Flip it.")
        else:
            desc.append("Do not flip it.")
    q = " ".join(desc) + " What side is it on now? Answer A or B."
    return q, state


def make_synthetic_data(task: str, n: int, seed: int) -> List[Tuple[str, int | str]]:
    random.seed(seed)
    data = []
    for _ in range(n):
        if task == "synthetic_arith":
            steps = random.randint(2, 5)
            data.append(gen_arith(steps=steps))
        elif task == "synthetic_coinflip":
            steps = random.randint(2, 6)
            data.append(gen_coinflip(n=steps))
        else:
            raise ValueError(f"Unsupported synthetic task: {task}")
    return data


def load_dataset(
    name: str,
    calibration_size: int,
    test_size: int,
    split_seed: int,
    cache_dir: str = ".cache/",
) -> Tuple[List[Tuple[str, int | str]], List[Tuple[str, int | str]]]:
    if name == "gsm8k":
        try:
            dataset = hf_load_dataset("gsm8k", "main", cache_dir=cache_dir)
        except Exception as exc:
            raise RuntimeError(f"Failed to load GSM8K dataset: {exc}") from exc
        train = dataset["train"].shuffle(seed=split_seed)
        test = dataset["test"].shuffle(seed=split_seed + 1)

        calib_size = min(calibration_size, len(train))
        test_size = min(test_size, len(test))

        calib = train.select(range(calib_size))
        test = test.select(range(test_size))

        def process(ex) -> Tuple[str, int]:
            question = ex["question"].strip()
            answer = extract_last_int(ex["answer"])
            if answer is None:
                raise ValueError("Failed to extract answer from GSM8K example")
            return question, answer

        calib_data = [process(ex) for ex in calib]
        test_data = [process(ex) for ex in test]
        return calib_data, test_data

    if name in {"synthetic_arith", "synthetic_coinflip"}:
        calib_data = make_synthetic_data(name, calibration_size, split_seed)
        test_data = make_synthetic_data(name, test_size, split_seed + 1)
        return calib_data, test_data

    raise ValueError(f"Unsupported dataset: {name}")
