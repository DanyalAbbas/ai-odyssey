"""
Leakage audit: check if training identities overlap with eval identities.
Uses embedding nearest-neighbor. Flag any pair with cosine > 0.5.
"""
import numpy as np
import argparse
import json


def cosine_sim(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def audit(train_emb, train_ids, eval_emb, eval_ids, threshold=0.5):
    sim = cosine_sim(train_emb, eval_emb)
    max_per_eval = sim.max(axis=0)
    flagged = max_per_eval > threshold

    results = {
        "num_train": len(train_ids),
        "num_eval": len(eval_ids),
        "num_flagged_eval": int(flagged.sum()),
        "max_cosine": float(max_per_eval.max()),
        "mean_max_cosine": float(max_per_eval.mean()),
        "threshold": threshold,
        "flagged_pairs": [],
    }

    for j in np.where(flagged)[0]:
        i = sim[:, j].argmax()
        results["flagged_pairs"].append({
            "train_id": str(train_ids[i]),
            "eval_id": str(eval_ids[j]),
            "cosine": float(sim[i, j]),
        })

    # Sort by cosine descending
    results["flagged_pairs"].sort(key=lambda x: -x["cosine"])
    results["flagged_pairs"] = results["flagged_pairs"][:50]  # top 50

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_emb", required=True, help="Path to .npy train embeddings (N x D)")
    parser.add_argument("--train_ids", required=True, help="Path to .npy train identity labels")
    parser.add_argument("--eval_emb", required=True, help="Path to .npy eval embeddings (M x D)")
    parser.add_argument("--eval_ids", required=True, help="Path to .npy eval identity labels")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="leakage_report.json")
    args = parser.parse_args()

    train_emb = np.load(args.train_emb)
    train_ids = np.load(args.train_ids, allow_pickle=True)
    eval_emb = np.load(args.eval_emb)
    eval_ids = np.load(args.eval_ids, allow_pickle=True)

    results = audit(train_emb, train_ids, eval_emb, eval_ids, args.threshold)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Train: {results['num_train']} embeddings")
    print(f"Eval:  {results['num_eval']} embeddings")
    print(f"Flagged eval identities: {results['num_flagged_eval']}")
    print(f"Max cosine: {results['max_cosine']:.4f}")
    print(f"Mean max cosine: {results['mean_max_cosine']:.4f}")

    if results["num_flagged_eval"] > 0:
        print(f"\nTop 10 flagged pairs:")
        for p in results["flagged_pairs"][:10]:
            print(f"  {p['train_id']} <-> {p['eval_id']}: {p['cosine']:.4f}")

    print(f"\nFull report saved to {args.output}")


if __name__ == "__main__":
    main()