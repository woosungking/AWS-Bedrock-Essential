from concurrent.futures import ThreadPoolExecutor
from typing import List
from util import llm_call # (extract_xml, Dict, Callable 등 안 쓰는 건 지워도 됩니다)

def parallel(prompt: str, inputs: List[str], n_workers: int = 3) -> List[str]:
    """동일한 프롬프트로 여러 입력을 동시에 처리할 수 있습니다."""
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(llm_call, f"{prompt}\nInput: {x}") for x in inputs]
        return [f.result() for f in futures]

# 꿀잼 예시: IT 회사 상황극 (스레드 3개가 동시에 3명의 반응을 상상합니다)
characters = [
    """5년차 백엔드 개발자:
    - 심각한 카페인 중독
    - "제 피씨에선 되는데요"가 입버릇
    - 영혼이 탈곡된 상태""",
    
    """입사 1주일 차 열정 신입:
    - 열정 과다, 모든 게 신기함
    - 서버 에러가 나면 다 자기 탓인 줄 알고 울먹임
    - 아직 깃헙(GitHub) PR 날리는 것도 버거움""",
    
    """무대뽀 기획자 (PM):
    - 개발은 1도 모름
    - "이거 버튼 하나만 추가하면 되는 거 아니에요?"가 단골 멘트
    - 일정은 무조건 내일 오픈"""
]

impact_results = parallel(
    """상황: 방금 "운영 서버(Production) DB 데이터가 전부 날아갔습니다!!" 라는 사내 메신저 공지가 올라왔습니다.
    
    지시사항:
    주어진 캐릭터의 성격이 200% 반영된 반응을 딱 '3줄짜리 짧은 독백이나 대사'로만 재미있게 써주세요.
    설명은 다 빼고 대사만 짧고 굵게 출력하세요.""",
    characters
)

print("\n🚨 [긴급 속보] 운영 서버 DB 증발 사태! 각 직군별 반응 🚨\n")
for i, result in enumerate(impact_results):
    print(f"[{i+1}번 스레드 완료]")
    print(result)
    print("-" * 40)
