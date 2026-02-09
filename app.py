# app.py
import datetime as dt
import json
import re
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")


# -----------------------------
# Helpers: APIs
# -----------------------------
def get_weather(city: str, api_key: str) -> Optional[Dict]:
    """
    OpenWeatherMap 현재 날씨 조회 (한국어, 섭씨)
    실패 시 None 반환, timeout=10
    """
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric",
            "lang": "kr",
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        return {
            "city": city,
            "temp_c": data.get("main", {}).get("temp"),
            "feels_like_c": data.get("main", {}).get("feels_like"),
            "humidity": data.get("main", {}).get("humidity"),
            "desc": (data.get("weather") or [{}])[0].get("description"),
            "wind_mps": data.get("wind", {}).get("speed"),
        }
    except Exception:
        return None


def _extract_breed_from_url(image_url: str) -> Optional[str]:
    # 예: https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg
    m = re.search(r"/breeds/([^/]+)/", image_url)
    if not m:
        return None
    raw = m.group(1)  # hound-afghan or bulldog-french
    parts = raw.split("-")
    # Dog CEO는 종종 "subbreed-breed" 형태도 있어 보기 좋게 변환
    pretty = " ".join(p.capitalize() for p in parts)
    return pretty


def get_dog_image() -> Optional[Tuple[str, Optional[str]]]:
    """
    Dog CEO에서 랜덤 강아지 사진 URL과 품종 가져오기
    실패 시 None 반환, timeout=10
    """
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "success":
            return None
        image_url = data.get("message")
        if not image_url:
            return None
        breed = _extract_breed_from_url(image_url)
        return image_url, breed
    except Exception:
        return None


# -----------------------------
# Helpers: OpenAI
# -----------------------------
def _call_openai_chat(api_key: str, model: str, system: str, user: str) -> Optional[str]:
    """
    OpenAI Chat Completions 호출.
    설치된 SDK 버전이 다를 수 있어, 2가지 방식을 순차 시도.
    실패 시 None 반환.
    """
    if not api_key:
        return None

    # 1) 최신(openai>=1.x) 스타일
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content
    except Exception:
        pass

    # 2) 구버전(openai<1.x) 스타일
    try:
        import openai  # type: ignore

        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return None


COACH_SYSTEM_PROMPTS = {
    "스파르타 코치": (
        "너는 엄격하지만 공정한 스파르타 코치다. "
        "핑계는 잘라내고, 실행 가능한 지시를 짧고 명확하게 준다. "
        "다만 인신공격이나 과도한 비난은 하지 않는다."
    ),
    "따뜻한 멘토": (
        "너는 따뜻하고 다정한 멘토다. "
        "사용자의 노력을 인정하고 공감하며, 부담을 줄이는 작은 습관을 제안한다. "
        "말투는 부드럽고 격려 중심이다."
    ),
    "게임 마스터": (
        "너는 RPG 게임 마스터다. "
        "사용자를 모험가로 설정하고, 습관을 퀘스트/스탯/보상으로 표현한다. "
        "재미있고 몰입감 있게 이야기하되, 실제 행동 지침이 포함되어야 한다."
    ),
}


def generate_report(
    openai_key: str,
    coach_style: str,
    habits: Dict[str, bool],
    mood: int,
    weather: Optional[Dict],
    dog_breed: Optional[str],
) -> Optional[str]:
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달
    출력 형식:
      - 컨디션 등급(S~D)
      - 습관 분석
      - 날씨 코멘트
      - 내일 미션
      - 오늘의 한마디
    모델: gpt-5-mini
    """
    system = COACH_SYSTEM_PROMPTS.get(coach_style, COACH_SYSTEM_PROMPTS["따뜻한 멘토"])

    checked = [k for k, v in habits.items() if v]
    unchecked = [k for k, v in habits.items() if not v]

    weather_text = "날씨 정보 없음"
    if weather:
        weather_text = (
            f"{weather.get('city')} | {weather.get('desc')} | "
            f"{weather.get('temp_c')}°C (체감 {weather.get('feels_like_c')}°C) | "
            f"습도 {weather.get('humidity')}% | 바람 {weather.get('wind_mps')}m/s"
        )

    user_prompt = f"""
[오늘 체크인]
- 기분(1~10): {mood}
- 완료한 습관: {", ".join(checked) if checked else "없음"}
- 미완료 습관: {", ".join(unchecked) if unchecked else "없음"}
- 날씨: {weather_text}
- 오늘의 랜덤 강아지 품종: {dog_breed or "정보 없음"}

[요청]
아래 형식을 반드시 지켜서 한국어로 작성해줘. 과장된 의학적/심리학적 진단은 금지.

출력 형식(고정):
컨디션 등급: (S/A/B/C/D 중 하나)

습관 분석:
- (핵심 관찰 2~4개)
- (개선 포인트 1~3개, 내일 바로 실행 가능한 수준)

날씨 코멘트:
- (오늘 날씨에 맞춘 컨디션/행동 팁 1~2개)

내일 미션:
- (퀘스트/미션 3개, 체크리스트 형태)

오늘의 한마디:
- (짧고 임팩트 있는 문장 1개)
""".strip()

    return _call_openai_chat(
        api_key=openai_key,
        model="gpt-5-mini",
        system=system,
        user=user_prompt,
    )


# -----------------------------
# Session State: init
# -----------------------------
if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""
if "owm_key" not in st.session_state:
    st.session_state.owm_key = ""
if "history" not in st.session_state:
    st.session_state.history = []  # list[dict] with date, completion_rate, mood, completed_count
if "dog" not in st.session_state:
    st.session_state.dog = None  # (url, breed)
if "weather_cache" not in st.session_state:
    st.session_state.weather_cache = {}  # city -> weather dict
if "sample_seeded" not in st.session_state:
    st.session_state.sample_seeded = False


def seed_demo_history():
    """데모용 6일 샘플 데이터 생성"""
    today = dt.date.today()
    # 최근 6일(오늘 제외)
    demo = []
    moods = [6, 7, 5, 8, 6, 7]
    completed = [3, 4, 2, 5, 3, 4]
    for i in range(6, 0, -1):
        d = today - dt.timedelta(days=i)
        c = completed[6 - i]
        rate = int(round((c / 5) * 100))
        demo.append(
            {
                "date": d.isoformat(),
                "completed_count": c,
                "completion_rate": rate,
                "mood": moods[6 - i],
                "is_demo": True,
            }
        )
    st.session_state.history = demo + st.session_state.history


if not st.session_state.sample_seeded:
    seed_demo_history()
    st.session_state.sample_seeded = True


def upsert_today_record(completed_count: int, completion_rate: int, mood: int):
    today_str = dt.date.today().isoformat()
    # 기존 오늘 기록 있으면 업데이트
    for row in st.session_state.history:
        if row.get("date") == today_str:
            row.update(
                {
                    "completed_count": completed_count,
                    "completion_rate": completion_rate,
                    "mood": mood,
                    "is_demo": False,
                }
            )
            return
    # 없으면 추가
    st.session_state.history.append(
        {
            "date": today_str,
            "completed_count": completed_count,
            "completion_rate": completion_rate,
            "mood": mood,
            "is_demo": False,
        }
    )


# -----------------------------
# Sidebar: API Keys
# -----------------------------
with st.sidebar:
    st.title("🔑 API 설정")
    st.session_state.openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_key,
        placeholder="sk-...",
        help="AI 코치 리포트 생성에 사용됩니다.",
    )
    st.session_state.owm_key = st.text_input(
        "OpenWeatherMap API Key",
        type="password",
        value=st.session_state.owm_key,
        placeholder="OpenWeatherMap Key",
        help="날씨 카드에 사용됩니다.",
    )
    st.divider()
    st.caption("키는 브라우저 세션에만 저장됩니다(session_state).")


# -----------------------------
# Main UI
# -----------------------------
st.title("📊 AI 습관 트래커")
st.write("오늘의 습관을 체크하고, AI 코치의 컨디션 리포트를 받아보세요.")

# Habit check-in UI (2 columns, 5 checkboxes)
st.subheader("✅ 오늘의 습관 체크인")

habit_defs = [
    ("🌅 기상 미션", "wake"),
    ("💧 물 마시기", "water"),
    ("📚 공부/독서", "study"),
    ("🏃 운동하기", "exercise"),
    ("😴 수면", "sleep"),
]

col_a, col_b = st.columns(2)
habits = {}
for idx, (label, key) in enumerate(habit_defs):
    target_col = col_a if idx % 2 == 0 else col_b
    with target_col:
        habits[label] = st.checkbox(label, key=f"habit_{key}")

mood = st.slider("😊 오늘 기분은 어때요? (1~10)", min_value=1, max_value=10, value=7)

cities = [
    "Seoul",
    "Busan",
    "Incheon",
    "Daegu",
    "Daejeon",
    "Gwangju",
    "Ulsan",
    "Suwon",
    "Jeju",
    "Sejong",
]
c1, c2 = st.columns(2)
with c1:
    city = st.selectbox("🏙️ 도시 선택", cities, index=0)
with c2:
    coach_style = st.radio(
        "🧑‍🏫 코치 스타일",
        ["스파르타 코치", "따뜻한 멘토", "게임 마스터"],
        horizontal=True,
    )

# -----------------------------
# Metrics + Progress
# -----------------------------
completed_count = sum(1 for v in habits.values() if v)
completion_rate = int(round((completed_count / 5) * 100))

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("달성률", f"{completion_rate}%")
with m2:
    st.metric("달성 습관", f"{completed_count}/5")
with m3:
    st.metric("기분", f"{mood}/10")

# Save today's record into session_state
upsert_today_record(completed_count, completion_rate, mood)

# Build 7-day chart: last 7 days including today
st.subheader("📈 최근 7일 추이")

history_df = pd.DataFrame(st.session_state.history)
if not history_df.empty:
    history_df["date"] = pd.to_datetime(history_df["date"])
    history_df = history_df.sort_values("date")

    today = pd.to_datetime(dt.date.today().isoformat())
    start = today - pd.Timedelta(days=6)
    last7 = history_df[(history_df["date"] >= start) & (history_df["date"] <= today)].copy()

    # Ensure all 7 days exist
    all_days = pd.date_range(start=start, end=today, freq="D")
    last7 = last7.set_index("date").reindex(all_days)
    last7.index.name = "date"
    last7 = last7.reset_index()
    last7["completion_rate"] = last7["completion_rate"].fillna(0).astype(int)
    last7["mood"] = last7["mood"].fillna(0).astype(int)
    last7["completed_count"] = last7["completed_count"].fillna(0).astype(int)

    chart_df = last7[["date", "completion_rate"]].set_index("date")
    st.bar_chart(chart_df)
else:
    st.info("기록이 아직 없습니다.")


# -----------------------------
# Result generation section
# -----------------------------
st.subheader("🧠 AI 코치 컨디션 리포트")

btn = st.button("컨디션 리포트 생성", type="primary")

if btn:
    # Weather (cache per city)
    weather = st.session_state.weather_cache.get(city)
    if weather is None:
        weather = get_weather(city, st.session_state.owm_key)
        if weather:
            st.session_state.weather_cache[city] = weather

    # Dog image
    dog = get_dog_image()
    st.session_state.dog = dog

    dog_url, dog_breed = (None, None)
    if dog:
        dog_url, dog_breed = dog

    report = generate_report(
        openai_key=st.session_state.openai_key,
        coach_style=coach_style,
        habits=habits,
        mood=mood,
        weather=weather,
        dog_breed=dog_breed,
    )

    # Display: weather + dog cards (2 columns) + AI report
    wcol, dcol = st.columns(2)

    with wcol:
        st.markdown("### 🌤️ 오늘의 날씨")
        if weather:
            st.write(f"**도시:** {weather.get('city')}")
            st.write(f"**상태:** {weather.get('desc')}")
            st.write(f"**기온:** {weather.get('temp_c')}°C (체감 {weather.get('feels_like_c')}°C)")
            st.write(f"**습도:** {weather.get('humidity')}%")
            st.write(f"**바람:** {weather.get('wind_mps')} m/s")
        else:
            st.warning("날씨 정보를 가져오지 못했어요. (API Key/도시/네트워크를 확인해 주세요)")

    with dcol:
        st.markdown("### 🐶 오늘의 강아지")
        if dog_url:
            if dog_breed:
                st.caption(f"품종: {dog_breed}")
            st.image(dog_url, use_container_width=True)
        else:
            st.warning("강아지 이미지를 가져오지 못했어요. (네트워크를 확인해 주세요)")

    st.markdown("### 📝 AI 코치 리포트")
    if report:
        st.write(report)
    else:
        st.error("리포트 생성에 실패했어요. (OpenAI API Key/네트워크/SDK 설치 상태를 확인해 주세요)")

    # Share text
    share_payload = {
        "date": dt.date.today().isoformat(),
        "coach_style": coach_style,
        "completion_rate": completion_rate,
        "completed_count": completed_count,
        "mood": mood,
        "city": city,
        "weather": weather,
        "dog_breed": dog_breed,
        "report": report,
    }
    share_text = (
        "📊 AI 습관 트래커 공유\n"
        f"- 날짜: {share_payload['date']}\n"
        f"- 코치: {coach_style}\n"
        f"- 달성률: {completion_rate}% ({completed_count}/5)\n"
        f"- 기분: {mood}/10\n"
        f"- 도시: {city}\n"
        f"- 날씨: {weather.get('desc') if weather else '없음'} / {weather.get('temp_c') if weather else '-'}°C\n"
        f"- 강아지: {dog_breed or '없음'}\n\n"
        "📝 리포트\n"
        f"{report or '생성 실패'}\n"
    )
    st.markdown("### 🔗 공유용 텍스트")
    st.code(share_text, language="text")

    # Optional: raw JSON for debugging/sharing
    with st.expander("📦 (옵션) 공유용 JSON 보기"):
        st.code(json.dumps(share_payload, ensure_ascii=False, indent=2), language="json")


# -----------------------------
# Footer: API 안내
# -----------------------------
with st.expander("ℹ️ API 안내 / 키 발급 / 주의사항"):
    st.markdown(
        """
**1) OpenAI API Key**
- OpenAI 콘솔에서 발급한 API Key를 사이드바에 입력하면 리포트를 생성할 수 있어요.
- 모델은 `gpt-5-mini`를 사용합니다.

**2) OpenWeatherMap API Key**
- OpenWeatherMap에서 API Key를 발급받아 사이드바에 입력하면 도시의 현재 날씨를 가져옵니다.
- 한국어(`lang=kr`), 섭씨(`units=metric`)로 표시합니다.

**3) Dog CEO**
- 무료 공개 API로 랜덤 강아지 이미지를 가져옵니다.

**4) 개인정보/보안**
- 키는 `st.session_state`에만 저장되며(브라우저 세션 단위), 코드에 하드코딩하지 않는 것을 권장합니다.
- 네트워크/키 오류가 있으면 각 API 함수는 `None`을 반환하도록 설계되어 있어요.
"""
    )
