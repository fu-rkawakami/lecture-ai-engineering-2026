"""第5回: ページ単位のベクトル検索。"""

import argparse
import json
import math
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from collections import Counter, defaultdict, deque
from pathlib import Path

from google import genai
from google.genai import types
from pydantic_ai.exceptions import ModelHTTPError
import sqlite_vec

from app.config import (
    EMBEDDING_DIMENSIONS, EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL,
    RAG_DB, RAG_DOCUMENT_IDS, RAG_SOURCE, RAG_TOP_K, RAG_TREE, get_api_key,
    RAG_GRAPH, GRAPH_START_K, GRAPH_HOPS, GRAPH_MAX_RESULTS,
)


def get_all_materials() -> str:
    """全資料を、講義・章・ページの情報付きで1つの文字列にまとめる。"""
    documents = json.loads(RAG_SOURCE.read_text(encoding="utf-8"))
    return "\n\n".join(
        f"資料: {document['id']} / {document['title']}\n"
        f"章: {section['section']}\n"
        f"ページ: PDF p.{page['page']} / {page['title']}\n{page['content']}"
        for document in documents
        for section in document["content"]
        for page in section["content"]
    )

def load_pages(source: Path = RAG_SOURCE) -> list[dict]:
    """資料→章→ページのJSONを、対象資料のページ一覧に変換する。"""
    documents = json.loads(source.read_text(encoding="utf-8"))
    pages = []
    for document in documents:
        if document["id"] not in RAG_DOCUMENT_IDS:
            continue
        for section in document["content"]:
            for page in section["content"]:
                pages.append({"page_id": page["id"], "document_id": document["id"],
                              "lecture": int(document["id"][3:5]), "document_title": document["title"],
                              "section": section["section"], "title": page["title"],
                              "page": page["page"], "content": page["content"]})
    return pages


def browse_tree(node_id: str | None = None) -> dict:
    """直下のページは先頭文、指定したページ自身は全文を返す。"""
    tree = json.loads(RAG_TREE.read_text(encoding="utf-8"))
    tree["children"] = [node for node in tree["children"] if node["id"] in RAG_DOCUMENT_IDS]
    node_id = node_id or tree["id"]
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}

    def describe(node, full=False):
        info = {key: node[key] for key in ("id", "kind", "title")}
        if node["kind"] == "page":
            info.update({key: value for key, value in pages[node["id"]].items() if key != "page_id"})
            if not full:
                info["content"] = re.split(r"(?<=[。！？])|\n", info["content"].strip(), maxsplit=1)[0]
        return info

    remaining = [tree]
    while remaining:
        node = remaining.pop()
        if node["id"] == node_id:
            return {
                "node": describe(node, full=True),
                "children": [describe(child) for child in node.get("children", [])],
            }
        remaining.extend(node.get("children", []))
    raise ValueError("対象の目次ノードが見つかりません。")


def read_page(page_id: str) -> dict:
    """第1〜3回の本講義から、指定した1ページの本文と引用情報を返す。"""
    for page in load_pages(RAG_SOURCE):
        if page["page_id"] == page_id:
            return page
    raise ValueError("対象のページが見つかりません。")


def normalized_vector(values: list[float]) -> list[float]:
    """ベクトルの長さを1に揃える。"""
    norm = math.hypot(*values)
    if norm == 0:
        raise ValueError("埋め込みがゼロベクトルです。")
    return [value / norm for value in values]


def embed_text(text: str, task_type: str) -> list[float]:
    """Geminiで文章を埋め込む。入力上限を超えた場合は切り捨てず停止する。"""
    with genai.Client(api_key=get_api_key()) as client:
        count = client.models.count_tokens(model=EMBEDDING_MODEL, contents=text).total_tokens
        if count is None or count > EMBEDDING_MAX_TOKENS:
            raise ValueError(f"埋め込み入力が上限を超えるか計測不能です: {count} tokens。")
        response = client.models.embed_content(
            model=EMBEDDING_MODEL, contents=text,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS),
        )
    embedding = response.embeddings[0]
    if embedding.statistics and embedding.statistics.truncated:
        raise ValueError("埋め込みAPIが本文を切り捨てました。")
    return normalized_vector(embedding.values)


@contextmanager
def open_index(path: Path):
    """既存の検索DBを読み取り専用で開き、終了時に閉じる。"""
    if not path.is_file():
        raise FileNotFoundError("検索DBがありません。")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        yield db
    finally:
        db.close()


def search(query: str) -> list[dict]:
    """質問を埋め込み、近いページを最大RAG_TOP_K件返す。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    with open_index(RAG_DB) as db:
        vector = embed_text(query.strip(), "RETRIEVAL_QUERY")
        rows = db.execute("""
            SELECT page_id, document_id, lecture, document_title, section, title, page, content,
                   vec_distance_cosine(embedding, ?) AS distance
            FROM chunks ORDER BY distance, page_id LIMIT ?
        """, (sqlite_vec.serialize_float32(vector), RAG_TOP_K))
        return [dict(row) for row in rows]


def _bm25_terms(text: str) -> list[str]:
    """英数字は単語、日本語は1文字・2文字の並びに分ける。"""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    terms = []
    for segment in re.findall(r"[a-z0-9]+|[^\W_a-z0-9]+", normalized):
        if segment.isascii():
            terms.append(segment)
        else:
            terms.extend(segment)
            terms.extend(segment[i:i + 2] for i in range(len(segment) - 1))
    return terms


def _bm25_page_ids(query: str, pages: dict[str, dict]) -> list[str]:
    """タイトルと本文でBM25を計算し、検索語が一致したページの上位を返す。"""
    query_terms = set(_bm25_terms(query))
    if not query_terms or not pages:
        return []

    page_terms = {
        page_id: Counter(_bm25_terms(f"{page['title']}\n{page['content']}"))
        for page_id, page in pages.items()
    }
    lengths = {page_id: terms.total() for page_id, terms in page_terms.items()}
    average_length = sum(lengths.values()) / len(lengths)
    if not average_length:
        return []
    document_frequency = Counter(
        term for terms in page_terms.values() for term in query_terms & terms.keys()
    )
    idf = {
        term: math.log1p((len(pages) - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }

    k1, b = 1.2, 0.75
    matched_terms = sorted(idf)
    scores = {}
    for page_id, terms in page_terms.items():
        normalization = k1 * (1 - b + b * lengths[page_id] / average_length)
        score = sum(
            idf[term] * terms[term] * (k1 + 1) / (terms[term] + normalization)
            for term in matched_terms
            if term in terms
        )
        if score > 0:
            scores[page_id] = score
    return sorted(scores, key=lambda page_id: (-scores[page_id], page_id))[:RAG_TOP_K]


def _rrf_graph_seeds(vector_hits: list[dict], bm25_ids: list[str], limit: int) -> list[dict]:
    """同じページの順位スコアを足し合わせ、RRF60の上位を入口にする。"""
    vector_ranks = {hit["page_id"]: rank for rank, hit in enumerate(vector_hits, 1)}
    bm25_ranks = {page_id: rank for rank, page_id in enumerate(bm25_ids, 1)}
    distances = {hit["page_id"]: hit["distance"] for hit in vector_hits}
    scores = defaultdict(float)
    for ranks in (vector_ranks, bm25_ranks):
        for page_id, rank in ranks.items():
            scores[page_id] += 1 / (60 + rank)

    page_ids = sorted(scores, key=lambda page_id: (-scores[page_id], page_id))[:limit]
    return [
        {
            "page_id": page_id,
            "distance": distances.get(page_id),
            "vector_rank": vector_ranks.get(page_id),
            "bm25_rank": bm25_ranks.get(page_id),
            "rrf_score": scores[page_id],
        }
        for page_id in page_ids
    ]


def search_graph(query: str, *, use_rrf: bool = False) -> dict:
    """検索で入口を選び、前後ページと共通用語で候補を広げる。"""
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]
             if node["kind"] == "term" or node["id"] in pages}
    adjacent = defaultdict(set)
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if edge["relation"] in ("next_page", "mentions") and a in nodes and b in nodes:
            adjacent[a].add(b)
            adjacent[b].add(a)

    vector_hits = [hit for hit in search(query) if hit["page_id"] in pages]
    seed_limit = min(GRAPH_START_K, GRAPH_MAX_RESULTS)
    if use_rrf:
        seeds = _rrf_graph_seeds(vector_hits, _bm25_page_ids(query, pages), seed_limit)
    else:
        seeds = [
            {"page_id": hit["page_id"], "distance": hit["distance"]}
            for hit in vector_hits[:seed_limit]
        ]
    seed_ids = {seed["page_id"] for seed in seeds}
    results, candidates = [], {}
    for rank, seed in enumerate(seeds):
        root = seed["page_id"]
        results.append({**pages[root], **seed, "hops": 0,
                        "path": [{"id": root, "title": pages[root]["title"]}]})
        queue = deque([(root, [root])])
        visited = {root}
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= GRAPH_HOPS:
                continue
            # 同じ深さなら、接続先が少ない用語からたどる。
            neighbors = sorted(adjacent[current], key=lambda n: (len(adjacent[n]), n))
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                route = path + [neighbor]
                queue.append((neighbor, route))
                if neighbor not in pages or neighbor in seed_ids:
                    continue
                frequency = max((len(adjacent[n]) for n in route if nodes[n]["kind"] == "term"), default=0)
                priority = (len(route) - 1, frequency, rank, neighbor)
                candidate = {**pages[neighbor], "hops": len(route) - 1,
                             "path": [{"id": n, "title": nodes[n]["title"]} for n in route]}
                if neighbor not in candidates or priority < candidates[neighbor][0]:
                    candidates[neighbor] = (priority, candidate)
    expanded = [candidate for _, candidate in sorted(candidates.values())]
    return {"seeds": results, "expanded": expanded[:max(0, GRAPH_MAX_RESULTS - len(results))],
            "max_hops": GRAPH_HOPS}



def run_full_context(agent, question: str, model, instructions: str, prompt_template: str):
    """全資料を一括投入し、検索ツールを使わずに回答する。"""
    prompt = prompt_template.format(materials=get_all_materials(), question=question)
    print(f"[full-context] 入力{len(prompt):,}文字")
    try:
        with agent.override(tools=[], instructions=instructions):
            return agent.run_sync(prompt, model=model)
    except ModelHTTPError as e:
        print(f"[full-context] APIエラー HTTP {e.status_code}: {e.body}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="エージェントを使わず、講義資料の検索だけを実行する")
    parser.add_argument("query", help="調べたい内容")
    args = parser.parse_args()
    for i, result in enumerate(search(args.query), 1):
        print(f"\n{'=' * 60}\n検索結果 {i}\n{'=' * 60}")
        print(f"講義: {result['document_title']}")
        print(f"章: {result['section']}")
        print(f"タイトル: {result['title']}")
        print(f"ページ: {result['page']}")
        print(f"\n{result['content']}")
