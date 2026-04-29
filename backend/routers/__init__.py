"""FastAPI 라우터 패키지 (운영웹통합명세서 §8).

각 라우터는 ``backend/main.py`` 의 ``app.include_router(...)`` 로 등록한다.
- ``chunks``  — GET/POST/PATCH/DELETE /api/chunks (Day 3+)
- ``tree``    — GET /api/tree (Day 3)
- ``preview`` — POST /api/preview/* (Day 5)
- ``indexing``— POST /api/indexing/* (Day 6)
"""
