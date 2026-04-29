# Onyx nginx — config 를 image 에 직접 베이크 (Windows Docker bind mount 우회).
# onyx+docmost 개발.md A2 풀 교체 측정용.

FROM nginx:1.25.5-alpine

RUN apk add --no-cache bash dos2unix

COPY app.conf.template app.conf.template.no-letsencrypt app.conf.template.prod \
     mcp.conf.inc.template mcp_upstream.conf.inc.template \
     run-nginx.sh \
     /nginx-templates/

# 빌드 시점에 CRLF → LF 변환. 런타임 sed 우회 (docker-compose YAML 의
# 백슬래시 손실 이슈 회피).
RUN dos2unix /nginx-templates/run-nginx.sh && chmod +x /nginx-templates/run-nginx.sh

CMD ["/bin/sh", "-c", "rm -f /etc/nginx/conf.d/default.conf && cp -a /nginx-templates/. /etc/nginx/conf.d/ && /etc/nginx/conf.d/run-nginx.sh app.conf.template"]
