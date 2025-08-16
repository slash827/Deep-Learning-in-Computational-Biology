#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def read_proteins_from_files(files: List[Path]) -> List[str]:
    proteins: List[str] = []
    for fp in files:
        if not fp.exists():
            continue
        with fp.open('r') as f:
            proteins.extend([line.strip() for line in f if line.strip()])
    return proteins


def chunk_list(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


@torch.no_grad()
def compute_embeddings(
    proteins: List[str],
    model_name: str = "Rostlab/prot_bert_bfd",
    pooling: str = "mean",
    device: str = "auto",
    batch_size: int = 32,
    max_length: int = 1024,
    fp16: bool = False,
) -> Dict[str, torch.Tensor]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device_t)
    if fp16 and device_t.type == "cuda":
        model.half()

    # ProtBert expects space-separated amino acids
    def tokenize(batch_seqs: List[str]):
        spaced = [" ".join(list(seq)) for seq in batch_seqs]
        return tokenizer(
            spaced,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    results: Dict[str, torch.Tensor] = {}
    for batch in tqdm(chunk_list(proteins, batch_size), desc="Embedding proteins"):
        inputs = tokenize(batch)
        inputs = {k: v.to(device_t) for k, v in inputs.items()}
        if fp16 and device_t.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # (B, L, H)

        if pooling == "cls":
            # CLS token is at position 0
            batch_emb = hidden[:, 0, :]
        else:
            # Mean over non-padding tokens (exclude padding via attention mask)
            mask = inputs["attention_mask"].unsqueeze(-1)  # (B, L, 1)
            summed = (hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            batch_emb = summed / lengths

        batch_emb = batch_emb.detach().cpu()
        for seq, emb in zip(batch, batch_emb):
            # Remove spaces to store with raw sequence key
            raw_seq = seq
            if " " in raw_seq:
                raw_seq = raw_seq.replace(" ", "")
            results[raw_seq] = emb

    return results


def main():
    parser = argparse.ArgumentParser(description="Cache ProteinBERT embeddings for proteins in dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory containing training_RBPs2.txt and test_RBPs2.txt")
    parser.add_argument("--out", type=str, required=True, help="Output path (.pt suggested)")
    parser.add_argument("--model", type=str, default="Rostlab/prot_bert_bfd", help="HuggingFace model name")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"], help="Pooling strategy over residues")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding computation")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|cuda")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length for tokenizer (lower is faster)")
    parser.add_argument("--fp16", action="store_true", help="Use float16 on CUDA for faster inference and lower memory")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    protein_files = [data_dir / "training_RBPs2.txt", data_dir / "test_RBPs2.txt"]
    proteins = read_proteins_from_files(protein_files)
    unique_proteins = sorted(list(set(proteins)))
    print(f"Found {len(unique_proteins)} unique proteins to embed")

    print(f"Model: {args.model}; Device: {args.device}; BS: {args.batch_size}; MaxLen: {args.max_length}; FP16: {args.fp16}")

    embeddings = compute_embeddings(
        proteins=unique_proteins,
        model_name=args.model,
        pooling=args.pooling,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        fp16=args.fp16,
    )

    # Save as a dict[str, tensor]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(embeddings, args.out)
    print(f"Saved embeddings for {len(embeddings)} proteins to {args.out}")


if __name__ == "__main__":
    main()


