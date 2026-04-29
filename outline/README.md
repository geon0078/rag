# Outline (운영자 wiki) — 분리된 Docker 스택

운영자용 페이지 편집기. 학생용 V4 RAG 의 데이터 master.

## 구성

- **outline** (port `3002`) — Outline web UI/API
- **outline-postgres** — Outline 전용 DB
- **outline-redis** — Outline 전용 캐시
- **outline-mailpit** (port `8025`) — magic link 캐쳐 (개발용)

## 실행

프로젝트 루트(`AI-ver-4/`)에서:

```bash
# 시작
docker compose -f outline/docker-compose.yml --env-file outline/.env --project-name outline up -d

# 중단 (volumes 보존)
docker compose -f outline/docker-compose.yml --project-name outline stop

# 완전 제거 (volumes 삭제 — 데이터 손실 주의)
docker compose -f outline/docker-compose.yml --project-name outline down -v

# 로그
docker compose -f outline/docker-compose.yml --project-name outline logs -f outline
```

> ⚠️ `--project-name outline` 을 항상 동일하게 지정해야 volume 이 재사용됨.

## 첫 로그인

1. http://localhost:3002 접속 → 이메일 입력
2. http://localhost:8025 (mailpit) → 매직 링크 클릭
3. 로그인 완료

운영용 SMTP 로 전환은 `.env` 의 `SMTP_HOST` 변경.

## 디렉토리 구조

```
outline/
├── docker-compose.yml   # Outline 스택 정의
├── .env                 # SECRET_KEY, UTILS_SECRET (gitignore 권장)
└── README.md            # 본 파일
```

## V4 RAG 와의 연동

Outline 페이지 변경 후 V4 인덱스 갱신 (수동, 1-2분):

```bash
# 프로젝트 루트에서
python scripts/index_outline.py
docker exec euljiu-redis redis-cli FLUSHDB  # 답변 캐시 비우기 (선택)
```

자동 동기화 (Outline webhook → V4) 는 향후 구현 예정.

## 백업

```bash
# DB 덤프
docker exec outline-postgres pg_dump -U outline outline > backups/outline_backup.sql

# Volumes (Docker 내부)
docker volume ls | grep outline
# outline_outline_pg_data, outline_outline_redis_data, outline_outline_data, outline_outline_mailpit_data
```

## 포트

| 서비스 | 호스트 포트 | 컨테이너 포트 | 용도 |
|---|---:|---:|---|
| outline | 3002 | 3000 | UI / API |
| outline-mailpit | 8025 | 8025 | 메일 확인 |
| outline-postgres | (내부) | 5432 | DB (외부 노출 X) |
| outline-redis | (내부) | 6379 | 캐시 (외부 노출 X) |

## 의존성

- **외부**: 없음 (자체 postgres/redis)
- **내부 (다른 컴포넌트와의 관계)**:
  - V4 backend (`euljiu-api`) 가 `outline:3002` API 호출 (host.docker.internal 경유)
  - 운영자가 V4 인덱싱 트리거 (수동)
