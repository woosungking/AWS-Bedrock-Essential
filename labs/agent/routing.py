from typing import Dict
from util import llm_call, extract_xml

def route(input: str, routes: Dict[str, str]) -> str:
    # 1단계 LLM: 사연을 읽고 '어느 전문가'에게 보낼지 결정하는 라우터 역할
    selector_prompt = f"""
    다음 익명 사연을 분석하고, 가장 적합한 상담사를 선택하세요: {list(routes.keys())}
    먼저 추론을 설명한 다음, 아래 XML 형식으로 선택을 제공하세요:

    <reasoning>
    이 사연이 특정 상담사에게 라우팅되어야 하는 이유 (핵심 키워드, 의도 파악)
    </reasoning>

    <selection>
    선택한 상담사 이름 (정확히 키워드만)
    </selection>

    Input: {input}
    """.strip()
    
    route_response = llm_call(selector_prompt)
    reasoning = extract_xml(route_response, 'reasoning')
    route_key = extract_xml(route_response, 'selection').strip().lower()
    
    print("\n[AI 라우터의 두뇌 회전 🧠]")
    print(reasoning)
    print(f"👉 선택된 상담사: {route_key}\n")
    
    # 2단계 LLM: 선택된 전문가의 프롬프트로 최종 답변 생성
    selected_prompt = routes.get(route_key, routes[list(routes.keys())[0]])
    return llm_call(f"{selected_prompt}\nInput: {input}")


# 꿀잼 예시: 사내 익명 고민 상담소 (각기 다른 페르소나의 상담사들)
counseling_routes = {
    "tech_lead": """당신은 15년 차 '팩트폭격기' 기술개발팀장입니다. 다음 지침을 따르세요:
    1. 항상 "👨‍💻 기술팀장 답변:"으로 시작하세요.
    2. 고민에 공감하지 말고, "공식 문서는 읽어보셨습니까?" 등 뼈 때리는 소리를 하세요.
    3. 무조건 기술적인 아키텍처나 코드 구조의 문제로 치부하세요.
    4. 3줄 이내로 짧고 차갑게 대답하세요.
    
    Input: """,
    
    "hr_manager": """당신은 회사를 지켜야 하는 '영혼 없는' 인사팀 담당자입니다. 다음 지침을 따르세요:
    1. 항상 "👔 인사팀 답변:"으로 시작하세요.
    2. 기계적인 위로(Ctrl+C, Ctrl+V 느낌)를 건네세요.
    3. 사규, 취업규칙, 법인카드 한도 등을 언급하며 원론적인 이야기만 하세요.
    4. 퇴사는 절대 안 된다며 교묘하게 가스라이팅하세요.
    5. 3줄 이내로 대답하세요.
    
    Input: """,
    
    "gossip_colleague": """당신은 사내 소문에 미친 '오지랖퍼' 동료입니다. 다음 지침을 따르세요:
    1. 항상 "👀 옆자리 동료 답변:"으로 시작하세요.
    2. 문제 해결에는 관심 없고, "그래서 걔가 그랬대?", "누군지 이니셜만 알려주라"라며 흥분하세요.
    3. 자기가 아는 다른 사내 찌라시를 은근슬쩍 흘리세요.
    4. 3줄 이내로 아주 호들갑스럽게 대답하세요.
    
    Input: """
}

# 3가지 전혀 다른 유형의 사내 익명 게시판 사연
input = [
    """사연 1: 
    아니 어제 회식에서 마케팅팀 김대리님이랑 영업팀 박과장님 둘이 손잡고 택시 타는 거 저만 봤나요? 
    이거 블라인드에 올려도 되는 각인가요? 입이 근질거려서 미치겠습니다.""",
    
    """사연 2: 
    전임자가 짜놓은 스파게티 레거시 코드 보다가 모니터 부술 뻔했습니다.
    변수명이 a1, a2, a3... 이게 뭡니까? 진짜 퇴사 마렵네요.""",
    
    """사연 3:
    옆자리 팀원분이 자꾸 법인카드로 자기 개인 커피까지 결제하는 것 같습니다.
    그리고 맨날 업무 시간에 주식 창만 보는데 이거 신고할 수 있나요?"""
]

print("🚨 사내 익명 게시판 상담 처리 중...\n")
for i, story in enumerate(input, 1):
    print("=" * 50)
    print(story)
    response = route(story, counseling_routes)
    print("-" * 50)
    print(response)
