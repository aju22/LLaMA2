from typing import List, Tuple, Dict, Optional, Union
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelConfig, Transformer


class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, config: ModelConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int,
              max_batch_size: int, device: str):

        prev_time = time.time()

        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoints found"
            chk_path = checkpoints[0]
            print(f"Loading checkpoint from {chk_path}")
            checkpoint = torch.load(chk_path, map_location="cpu")
            print("Loaded Checkpoint in {:.2f}s".format(time.time() - prev_time))
            prev_time = time.time()

        with open(Path(checkpoints_dir) / "config.json", "r") as f:
            config = json.load(f.read())

        model_config = ModelConfig(max_seq_len=max_seq_len, max_batch_size=max_batch_size, device=device, **config)

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)

        model_config.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)

        model = Transformer(model_config).to(device)

        if load_model:
            model.load_state_dict(checkpoint, strict=True)
            print("Loaded Model in {:.2f}s".format(time.time() - prev_time))

        return LLaMA(model, tokenizer, model_config)

    @staticmethod
    def _sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:

        probs_sort, probs_indices = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0
        probs_sort /= probs_sort.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = probs_indices.gather(dim=-1, index=next_token)

        return next_token

    def generate(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9,
                 max_tokens: Optional[int] = None):

        if max_tokens is None:
            max_tokens = self.config.max_seq_len - 1

        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]

        batch_size = len(prompt_tokens)
        assert batch_size <= self.config.max_batch_size, f"Batch size {batch_size} exceeds max batch size {self.config.max_batch_size}"

        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.config.max_seq_len, f"Prompt length {max_prompt_len} exceeds max sequence length {self.config.max_seq_len}"

        total_len = min(self.config.max_seq_len, max_prompt_len + max_tokens)

        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.config.device)

        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.config.device)

        eos_reached = torch.Tensor([False] * batch_size).to(self.config.device)
        prompt_tokens_mask = tokens != pad_id

        for cur_pos in tqdm(range(1, total_len), desc="Generating Tokens.."):
            with torch.no_grad():
                logits = self.model(tokens[:, cur_pos - 1:cur_pos], cur_pos)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)

            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())

            if all(eos_reached):
                break

        out_tokens = []
        out_text = []

        for prompt_index, current_prompt_token in enumerate(tokens.tolist()):

            if self.tokenizer.eos_id() in current_prompt_token:
                eos_idx = current_prompt_token.index(self.tokenizer.eos_id())
                current_prompt_token = current_prompt_token[:eos_idx]

            out_tokens.append(current_prompt_token)
            out_text.append(self.tokenizer.decode(current_prompt_token))

        return out_tokens, out_text


if __name__ == '__main__':
    torch.manual_seed(0)

    allow_cuda = False
    device = "cuda" if allow_cuda and torch.cuda.is_available() else "cpu"

    model = LLaMA.build(
        checkpoints_dir="Llama-7b/checkpoints",
        tokenizer_path="Llama-7b/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=8,
        device=device
    )

    prompts = [
        "Who is the best football player in the world?"
    ]

    tokens, text = model.generate(prompts, temperature=0.6, top_p=0.9, max_tokens=32)
    assert len(tokens) == len(text) == len(prompts)

    for prompt, token, txt in zip(prompts, tokens, text):
        print(f"Prompt: {prompt}")
        print(f"Tokens: {token}")
        print(f"Text: {txt}")
        print('-' * 50)
