"""Solar OCR endpoint (Upstage Document Parse).

Vue frontend 의 파일 업로드를 받아 Upstage Document Parse API 로 forward.
지원 파일: 이미지 (jpg/png/heic), PDF, DOCX, PPTX, XLSX, HWP 등.

Endpoints:
  POST /api/ocr               — 파일 업로드 → 텍스트/마크다운/HTML 추출
  POST /api/ocr/extract-text  — 위와 동일하지만 plain text 만 추출 (편의용)

설정:
  UPSTAGE_API_KEY  — env 또는 /workspace/.env 에서 자동 로드
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/api/ocr", tags=["ocr"])

UPSTAGE_OCR_URL = "https://api.upstage.ai/v1/document-digitization"
DEFAULT_MODEL = "document-parse"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Upstage Document Parse 가 받는 확장자
ALLOWED_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp",
    ".pdf", ".docx", ".pptx", ".xlsx", ".hwp", ".hwpx",
}


def _api_key() -> str:
    key = os.environ.get("UPSTAGE_API_KEY", "")
    if not key:
        env_path = Path("/workspace/.env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("UPSTAGE_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise HTTPException(503, detail="UPSTAGE_API_KEY 미설정")
    return key


def _formats_to_payload(output_formats: str) -> str:
    """csv 'markdown,text,html' → JSON array string '["markdown","text","html"]'."""
    items = [f.strip() for f in output_formats.split(",") if f.strip()]
    return "[\"" + "\",\"".join(items) + "\"]" if items else '["markdown"]'


@router.post("")
async def ocr(
    document: UploadFile = File(...),
    model: str = Form(default=DEFAULT_MODEL),
    output_formats: str = Form(default="markdown,text,html"),
):
    """문서 OCR — 마크다운/텍스트/HTML 모두 반환."""
    ext = Path(document.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            400,
            detail=f"지원하지 않는 파일 형식: {ext}. 지원: {sorted(ALLOWED_EXTS)}",
        )

    content = await document.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, detail=f"파일 크기 50MB 초과: {len(content)} bytes")
    if not content:
        raise HTTPException(400, detail="빈 파일")

    files = {"document": (document.filename, content, document.content_type)}
    data = {
        "model": model,
        "output_formats": _formats_to_payload(output_formats),
        "ocr": "auto",
    }
    headers = {"Authorization": f"Bearer {_api_key()}"}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(UPSTAGE_OCR_URL, headers=headers, files=files, data=data)
    except httpx.RequestError as exc:
        raise HTTPException(502, detail=f"Upstage 호출 실패: {exc}") from exc

    if r.status_code != 200:
        raise HTTPException(
            r.status_code,
            detail=f"Upstage OCR 오류: {r.text[:500]}",
        )

    result = r.json()
    return {
        "filename": document.filename,
        "model": model,
        "content": result.get("content") or {},
        "elements": result.get("elements") or [],
        "usage": result.get("usage") or {},
    }


@router.post("/extract-text")
async def ocr_extract_text(
    document: UploadFile = File(...),
    model: str = Form(default=DEFAULT_MODEL),
):
    """OCR — plain text 만 반환 (간단 version)."""
    ext = Path(document.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, detail=f"지원하지 않는 파일 형식: {ext}")

    content = await document.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, detail="파일 크기 50MB 초과")
    if not content:
        raise HTTPException(400, detail="빈 파일")

    files = {"document": (document.filename, content, document.content_type)}
    data = {"model": model, "output_formats": '["text"]', "ocr": "auto"}
    headers = {"Authorization": f"Bearer {_api_key()}"}
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(UPSTAGE_OCR_URL, headers=headers, files=files, data=data)
    if r.status_code != 200:
        raise HTTPException(r.status_code, detail=r.text[:500])
    return {
        "filename": document.filename,
        "text": (r.json().get("content") or {}).get("text", ""),
    }
