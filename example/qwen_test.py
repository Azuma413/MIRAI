"""Qwen3.5-2B の動作確認スクリプト

transformers を使って Qwen/Qwen3.5-2B を読み込み，
テキスト生成の動作確認を行います．
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# リポジトリルートの weights/ フォルダにモデルをダウンロード
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def main():
    model_name = "Qwen/Qwen3.5-2B"
    WEIGHTS_DIR.mkdir(exist_ok=True)
    print(f"モデルの読み込み中: {model_name}")
    print(f"キャッシュディレクトリ: {WEIGHTS_DIR}")

    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=WEIGHTS_DIR)

    # モデルの読み込み（GPU が利用可能なら GPU を使用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=WEIGHTS_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    print(f"デバイス: {device}")
    print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # テスト用プロンプト
    prompts = [
        "The future of artificial intelligence is",
        "Explain quantum computing in simple terms:",
        "Write a Python function that calculates the Fibonacci sequence:",
    ]

    for prompt in prompts:
        print(f"=== プロンプト ===\n{prompt}\n")

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
            )

        # 入力部分を除いた生成テキストのみをデコード
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"=== 生成結果 ===\n{response}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
