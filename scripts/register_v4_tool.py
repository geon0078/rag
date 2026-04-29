"""Onyx 에 V4 RAG proxy 를 custom tool 로 등록.

V4 의 /api/onyx/search 를 호출하는 OpenAPI 3.0 스키마를 Onyx 에 POST.
이후 persona__tool 에서 internal_search 제거 + 새 tool 추가.

요구:
  - V4 (euljiu-api) 가 host.docker.internal:8000 으로 도달 가능 (확인됨).
  - Onyx admin API 토큰.

Run:
    ONYX_API_KEY=on_... python scripts/register_v4_tool.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ONYX_BASE = os.environ.get("ONYX_BASE_URL", "http://localhost:3010")

OPENAPI_SCHEMA = {
    "openapi": "3.0.0",
    "info": {
        "title": "을지대 학사 자료 검색 (V4 RAG)",
        "version": "1.0.0",
        "description": (
            "을지대학교 corpus (학칙·강의평가·시설·장학금·FAQ 등) 한국어 RAG 검색. "
            "Solar embedding 4096-dim + KoNLPy Okt BM25 hybrid + reranker."
        ),
    },
    "servers": [
        {"url": "http://host.docker.internal:8000"},
    ],
    "paths": {
        "/api/onyx/search": {
            "post": {
                "operationId": "search_eulji_corpus",
                "summary": "을지대 학사 corpus 검색",
                "description": (
                    "사용자 질문에 답하기 위한 을지대 corpus 를 검색합니다. "
                    "학과·시설·학사일정·장학금·학칙·강의평가 등 모든 학사 관련 질문에 "
                    "이 도구를 사용하세요."
                ),
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["query"],
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "사용자 질문 (한국어)",
                                    },
                                    "top_k": {
                                        "type": "integer",
                                        "default": 8,
                                        "description": "반환할 문서 수 (1-15)",
                                    },
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "검색 결과",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string"},
                                        "results": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "document_id": {"type": "string"},
                                                    "semantic_identifier": {"type": "string"},
                                                    "blurb": {"type": "string"},
                                                    "source_type": {"type": "string"},
                                                    "link": {"type": "string"},
                                                    "score": {"type": "number"},
                                                },
                                            },
                                        },
                                    },
                                }
                            }
                        },
                    }
                },
            }
        }
    },
}


def _hdr() -> dict[str, str]:
    key = os.environ.get("ONYX_API_KEY", "")
    if not key:
        raise SystemExit("ONYX_API_KEY 필요")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def main() -> int:
    body = {
        "name": "search_eulji_corpus",
        "description": (
            "을지대학교 학사 자료 (학칙·강의평가·시설·장학금·FAQ·학사일정 등) 한국어 검색. "
            "모든 을지대 관련 질문에 사용하세요. V4 RAG (Solar 4096 + Okt BM25 + reranker)."
        ),
        "definition": OPENAPI_SCHEMA,
        "custom_headers": [],
        "passthrough_auth": False,
    }
    with httpx.Client(timeout=30.0) as client:
        r = client.post(
            f"{ONYX_BASE}/api/admin/tool/custom",
            headers=_hdr(),
            json=body,
        )
        if r.status_code not in (200, 201):
            print(f"create custom tool 실패 {r.status_code}: {r.text[:500]}")
            return 1
        tool = r.json()
        print(f"[register] tool created id={tool.get('id')} name={tool.get('name')}")
        out_path = PROJECT_ROOT / "tmp" / "onyx_v4_tool.json"
        out_path.write_text(json.dumps(tool, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[register] saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
