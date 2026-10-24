"""第1回: 会話履歴を保ちながらLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --once "LoRAとは何ですか？"  # 1回だけ質問して終了

回答後に続けて質問を入力できる。/exit または /quit で終了する。
会話履歴は実行中だけ保持し、終了すると消える。
ツールは持たないため、
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from time import perf_counter

from google import genai
from google.genai import errors
from google.genai.chats import Chat

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, chat: Chat) -> str:
    # 同じチャットに送ることで、SDKが過去の質問・回答も一緒に送信する。
    start = perf_counter()
    response = chat.send_message(question)
    elapsed = perf_counter() - start
    # 通信やSDK内の処理も含め、回答全体が返るまでの時間。
    print(f"[応答時間] {elapsed:.3f} 秒")

    usage = response.usage_metadata
    if usage is not None:
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        print(f"[入力トークン数] {input_tokens if input_tokens is not None else '不明'}")
        print(f"[出力トークン数] {output_tokens if output_tokens is not None else '不明'}")
        thought_tokens = usage.thoughts_token_count
        total_tokens = usage.total_token_count
        print(f"[思考トークン数] {thought_tokens if thought_tokens is not None else '不明'}")
        print(f"[合計トークン数] {total_tokens if total_tokens is not None else '不明'}\n")

    return response.text or "（テキストの回答がありませんでした）"


def main() -> None:
    parser = argparse.ArgumentParser(description="会話履歴を保ちながらLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--once", action="store_true", help="1回だけ質問して終了する（比較計測用）")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if not args.once:
        print("続けて質問できます。終了するには /exit または /quit を入力してください。\n")
    try:
        with genai.Client(api_key=get_api_key()) as client:
            # チャットはループの外で1回だけ作る。
            chat = client.chats.create(model=MODEL)
            question = args.question
            while True:
                print(f"[質問] {question}\n")
                print(ask(question, chat))
                if args.once:
                    break
                while True:
                    question = input("\n[次の質問] ").strip()
                    if question:
                        break
                if question.lower() in ("/exit", "/quit"):
                    break
    except (EOFError, KeyboardInterrupt):
        print("\n会話を終了します。")
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
