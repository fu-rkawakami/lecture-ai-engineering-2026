"""BM25を加えたGraph検索とCLIの配線を、APIを呼ばずに確認する。"""

import json

import pytest
from pydantic_ai.models.test import TestModel

import app.agent as agent_module
from app import retrieval


def test_graph_rrf_combines_rankings_and_expands_from_bm25_seed(tmp_path, monkeypatch):
    pages = [
        {"page_id": "p1", "title": "概要", "content": "検索方式の概要"},
        {"page_id": "p2", "title": "関連資料", "content": "ベクトル検索の例"},
        {"page_id": "p3", "title": "用語一覧", "content": "OpenClaw " + "他の用語の説明 " * 30},
        {"page_id": "p4", "title": "OpenClaw", "content": "日々のメモ"},
        {"page_id": "p5", "title": "長期メモリ", "content": "選んだ記録を読み戻す"},
    ]
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps({
        "nodes": [
            {"id": page["page_id"], "kind": "page", "title": page["title"]} for page in pages
        ],
        "edges": [{"source": "p4", "target": "p5", "relation": "next_page"}],
    }), encoding="utf-8")
    monkeypatch.setattr(retrieval, "RAG_GRAPH", graph)
    monkeypatch.setattr(retrieval, "load_pages", lambda _: pages)
    vector_hits = [{"page_id": page_id, "distance": rank / 10}
                   for rank, page_id in enumerate(("p1", "p2", "p3"), 1)]
    monkeypatch.setattr(retrieval, "search", lambda _: vector_hits)

    baseline = retrieval.search_graph("ＯｐｅｎＣｌａｗ")
    assert [seed["page_id"] for seed in baseline["seeds"]] == ["p1", "p2", "p3"]
    assert baseline["expanded"] == []

    result = retrieval.search_graph("ＯｐｅｎＣｌａｗ", use_rrf=True)
    assert [seed["page_id"] for seed in result["seeds"]] == ["p3", "p1", "p4"]
    shared, _, lexical = result["seeds"]
    assert shared["rrf_score"] == pytest.approx(1 / 63 + 1 / 62)
    assert lexical["vector_rank"] is None and lexical["bm25_rank"] == 1
    assert lexical["rrf_score"] == pytest.approx(1 / 61)
    assert [page["page_id"] for page in result["expanded"]] == ["p5"]
    assert [step["id"] for step in result["expanded"][0]["path"]] == ["p4", "p5"]

    no_lexical_match = retrieval.search_graph("zzzz", use_rrf=True)
    assert [seed["page_id"] for seed in no_lexical_match["seeds"]] == ["p1", "p2", "p3"]


def test_graph_rrf_cli_calls_search_graph_and_prints_ranks(monkeypatch, capsys):
    calls = []

    def search_graph(query, *, use_rrf):
        calls.append(use_rrf)
        return {
            "seeds": [
                {"page_id": "p1", "title": "入口", "vector_rank": 1,
                 "bm25_rank": 2, "rrf_score": 1 / 61 + 1 / 62},
            ],
            "expanded": [
                {"title": "関連ページ", "hops": 1, "path": [{"id": "p1"}, {"id": "p2"}]},
            ],
        }

    model = TestModel(call_tools=["search_graph"])
    monkeypatch.setattr(retrieval, "search_graph", search_graph)
    monkeypatch.setattr(agent_module, "build_model", lambda: model)
    monkeypatch.setattr("sys.argv", ["app.agent", "--graph-rrf", "--trace", "メモリとは？"])

    agent_module.main()

    assert calls == [True]
    offered = model.last_model_request_parameters.function_tools
    assert [tool.name for tool in offered] == ["search_graph"]
    assert set(offered[0].parameters_json_schema["properties"]) == {"query"}
    output = capsys.readouterr().out
    assert "ベクトル順位 1、BM25順位 2" in output
    assert "[追加 1 hop] p1 → p2" in output
