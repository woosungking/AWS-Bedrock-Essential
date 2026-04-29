# AWS Bedrock Essential

Amazon Bedrock 핵심 기능 실습 워크샵입니다.

## 폴더 구조

```
labs/
├── agent/              # AI 에이전트 패턴 (prompt chaining, routing, parallelization)
├── api/                # Bedrock API 기본 호출
├── chatbot/            # 챗봇 구현
├── converse/           # Converse API
├── csv/                # CSV 데이터 처리
├── embedding/          # 텍스트 임베딩
├── embeddings_search/  # 임베딩 기반 검색
├── image/              # 이미지 생성
├── image_background/   # 배경 변경
├── image_extension/    # 이미지 확장
├── image_insertion/    # 이미지 삽입
├── image_masking/      # 이미지 마스킹
├── image_prompts/      # 이미지 프롬프트
├── image_replacement/  # 이미지 교체
├── image_search/       # 이미지 검색
├── image_style_mixing/ # 스타일 믹싱
├── image_to_image/     # 이미지 변환
├── image_understanding/# 이미지 이해
├── image_variation/    # 이미지 변형
├── intro_streaming/    # 스트리밍 입문
├── json/               # JSON 출력 제어
├── langchain/          # LangChain 연동
├── model_comparison/   # 모델 비교
├── multimodal_chatbot/ # 멀티모달 챗봇
├── params/             # 파라미터 튜닝
├── rag/                # RAG (검색 증강 생성)
├── rag_chatbot/        # RAG 챗봇
├── recommendations/    # 추천 시스템
├── showcase/           # 종합 쇼케이스
├── similarity/         # 유사도 검색
├── simple_streamlit/   # Streamlit 기본
├── streaming/          # 스트리밍 응답
├── summarization/      # 문서 요약
├── temperature/        # Temperature 실험
├── templates/          # 프롬프트 템플릿
├── text/               # 텍스트 생성
├── text_playground/    # 텍스트 플레이그라운드
└── tool_use/           # Tool Use (Function Calling)
```

## 시작하기

```bash
# 가상환경 활성화
source .venv/bin/activate

# 패키지 설치
pip install -r setup/requirements.txt
```

## 사전 요구사항

- Python 3.9+
- AWS 계정 및 Bedrock 모델 접근 권한
- AWS CLI 설정 완료
