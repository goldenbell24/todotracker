# app.py
import datetime as dt
import json
import re
from typing import Dict, Optional, Tuple, List

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
    """OpenWeatherMap 현재 날씨 조회 (한국어, 섭씨) / 실패 시 None / timeout=10"""
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric", "lang": "kr"}
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
    m = re.search(r"/breeds/([^/]+)/", image_url)
    if not m:
        return None
    raw = m.group(1)
    parts = raw.split("-")
    return " ".join(p.capitalize() for p in parts)


def get_dog_image() -> Optional[Tuple[str, Optional[str]]]:
    """Dog CEO 랜덤 강아지 이미지 URL + 품종 / 실패 시 None / timeout=10"""
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
    """OpenAI Chat 호출 (SDK 버전 차이 대비). 실패 시 None."""
    if not api_key:
        return None

    # openai>=1.x
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content
    except Exception:
        pass

    # openai<1.x
    try:
        import openai  # type: ignore

        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
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
    mood_label: str,
    weather: Optional[Dict],
    dog_breed: Optional[str],
) -> Optional[str]:
    """
    습관+기분+날씨+강아지 품종 -> OpenAI 전달
    출력 형식 고정, 모델: gpt-5-mini
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
- 기분(1~10): {mood} ({mood_label})
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
# Mood labels
# -----------------------------
def mood_descriptor(score: int) -> Tuple[str, str]:
    """
    1~10 점수를 짧은 표현 + 이모지로 매핑
    """
    if score <= 2:
        return "😣 많이 지침/우울", "😣"
    if score <= 4:
        return "😕 컨디션 저하", "😕"
    if score <= 6:
        return "🙂 무난/보통", "🙂"
    if score <= 8:
        return "😄 좋음/상승세", "😄"
    return "🤩 최고조/아주 좋음", "🤩"


# -----------------------------
# Session State: init
# -----------------------------
if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""
if "owm_key" not in st.session_state:
    st.session_state.owm_key = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "dog" not in st.session_state:
    st.session_state.dog = None
if "weather_cache" not in st.session_state:
    st.session_state.weather_cache = {}
if "sample_seeded" not in st.session_state:
    st.session_state.sample_seeded = False

# ✅ 습관 목록을 session_state로 관리 (고정 X)
if "habits_list" not in st.session_state:
    st.session_state.habits_list = [
        {"name": "기상 미션", "emoji": "🌅"},
        {"name": "물 마시기", "emoji": "💧"},
        {"name": "공부/독서", "emoji": "📚"},
        {"name": "운동하기", "emoji": "🏃"},
        {"name": "수면", "emoji": "😴"},
    ]


def seed_demo_history():
    """데모용 6일 샘플 데이터"""
    today = dt.date.today()
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


def upsert_today_record(completed_count: int, completion_rate: int, mood: int, total_habits: int):
    """오늘 기록 저장/업데이트 (총 습관 수가 변할 수 있어 같이 저장)"""
    today_str = dt.date.today().isoformat()
    for row in st.session_state.history:
        if row.get("date") == today_str:
            row.update(
                {
                    "completed_count": completed_count,
                    "completion_rate": completion_rate,
                    "mood": mood,
                    "total_habits": total_habits,
                    "is_demo": False,
                }
            )
            return
    st.session_state.history.append(
        {
            "date": today_str,
            "completed_count": completed_count,
            "completion_rate": completion_rate,
            "mood": mood,
            "total_habits": total_habits,
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

st.subheader("✅ 오늘의 습관 체크인")

# -----------------------------
# 1) 습관 편집 버튼 + 편집 UI
# -----------------------------
top_left, top_right = st.columns([3, 1])
with top_right:
    edit_mode = st.toggle("✏️ 편집", value=False, help="습관을 추가/삭제/이름 수정할 수 있어요.")

if edit_mode:
    st.info("습관 목록을 수정한 뒤 **저장**을 누르세요. (이름은 중복되지 않게 추천)")

    # 편집용 임시 리스트 (세션에 바로 반영하지 않고 저장 버튼에서 반영)
    if "habits_draft" not in st.session_state:
        st.session_state.habits_draft = [h.copy() for h in st.session_state.habits_list]

    draft: List[Dict] = st.session_state.habits_draft

    # 목록 편집
    for i, h in enumerate(draft):
        c1, c2, c3 = st.columns([1, 4, 1])
        with c1:
            emoji = st.text_input("이모지", value=h.get("emoji", "✅"), key=f"draft_emoji_{i}")
        with c2:
            name = st.text_input("습관 이름", value=h.get("name", ""), key=f"draft_name_{i}")
        with c3:
            remove = st.button("🗑️", key=f"remove_{i}")
        # 즉시 반영
        h["emoji"] = (emoji or "✅").strip()
        h["name"] = (name or "").strip()

        if remove:
            draft.pop(i)
            st.rerun()

    st.divider()
    add_c1, add_c2, add_c3 = st.columns([1, 4, 1])
    with add_c1:
        new_emoji = st.text_input("새 이모지", value="✨", key="new_habit_emoji")
    with add_c2:
        new_name = st.text_input("새 습관 이름", value="", key="new_habit_name")
    with add_c3:
        if st.button("➕ 추가"):
            n = (new_name or "").strip()
            e = (new_emoji or "✨").strip()
            if n:
                draft.append({"name": n, "emoji": e})
                st.session_state.new_habit_name = ""
                st.rerun()
            else:
                st.warning("새 습관 이름을 입력해 주세요.")

    save_c1, save_c2 = st.columns([1, 5])
    with save_c1:
        if st.button("💾 저장", type="primary"):
            # 유효성: 빈 이름 제거, 중복 이름 처리(중복이면 뒤에 (2) 붙임)
            cleaned = []
            seen = {}
            for h in draft:
                name = (h.get("name") or "").strip()
                if not name:
                    continue
                emoji = (h.get("emoji") or "✅").strip() or "✅"
                base = name
                if base in seen:
                    seen[base] += 1
                    name = f"{base} ({seen[base]})"
                else:
                    seen[base] = 1
                cleaned.append({"name": name, "emoji": emoji})

            if not cleaned:
                st.error("최소 1개 이상의 습관이 필요해요.")
            else:
                st.session_state.habits_list = cleaned
                # draft 재생성
                st.session_state.habits_draft = [h.copy() for h in cleaned]
                # 기존 체크박스 키 충돌 방지: 체크 키에 버전 사용
                st.session_state.habits_version = st.session_state.get("habits_version", 0) + 1
                st.success("습관 목록을 저장했어요!")
                st.rerun()

    with save_c2:
        if st.button("↩️ 변경 취소"):
            st.session_state.habits_draft = [h.copy() for h in st.session_state.habits_list]
            st.rerun()

else:
    # 편집 모드 종료 시 draft는 최신으로 맞춰둠
    st.session_state.habits_draft = [h.copy() for h in st.session_state.habits_list]


# -----------------------------
# 습관 체크박스 UI (2열) - 동적 생성
# -----------------------------
habits_list = st.session_state.habits_list
habits_version = st.session_state.get("habits_version", 0)

col_a, col_b = st.columns(2)
habits_checked: Dict[str, bool] = {}

for idx, h in enumerate(habits_list):
    emoji = h.get("emoji", "✅")
    name = h.get("name", f"습관 {idx+1}")
    label = f"{emoji} {name}".strip()

    # key는 habits_version 포함해서, 편집 후 위젯 상태/충돌 최소화
    widget_key = f"habit_{habits_version}_{idx}_{name}"
    target_col = col_a if idx % 2 == 0 else col_b
    with target_col:
        habits_checked[label] = st.checkbox(label, key=widget_key)

# -----------------------------
# 2) 기분 슬라이더: 구간별 라벨 + 이모지
# -----------------------------
mood = st.slider("😊 오늘 기분은 어때요? (1~10)", min_value=1, max_value=10, value=7)
mood_label, mood_emoji = mood_descriptor(mood)
st.caption(f"{mood_emoji} **기분 해석:** {mood_label}")

cities = ["Seoul", "Busan", "Incheon", "Daegu", "Daejeon", "Gwangju", "Ulsan", "Suwon", "Jeju", "Sejong"]
c1, c2 = st.columns(2)
with c1:
    city = st.selectbox("🏙️ 도시 선택", cities, index=0)
with c2:
    coach_style = st.radio("🧑‍🏫 코치 스타일", ["스파르타 코치", "따뜻한 멘토", "게임 마스터"], horizontal=True)

# -----------------------------
# Metrics + Progress (총 습관 수 가변)
# -----------------------------
total_habits = max(1, len(habits_list))
completed_count = sum(1 for v in habits_checked.values() if v)
completion_rate = int(round((completed_count / total_habits) * 100))

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("달성률", f"{completion_rate}%")
with m2:
    st.metric("달성 습관", f"{completed_count}/{total_habits}")
with m3:
    st.metric("기분", f"{mood}/10")

# Save today's record
upsert_today_record(completed_count, completion_rate, mood, total_habits)

# -----------------------------
# Chart (최근 7일)
# -----------------------------
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

    st.bar_chart(last7.set_index("date")[["completion_rate"]])
else:
    st.info("기록이 아직 없습니다.")


# -----------------------------
# AI Report Section
# -----------------------------
st.subheader("🧠 AI 코치 컨디션 리포트")
btn = st.button("컨디션 리포트 생성", type="primary")

if btn:
    # Weather cache by city
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
        habits=habits_checked,
        mood=mood,
        mood_label=mood_label,
        weather=weather,
        dog_breed=dog_breed,
    )

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

    share_payload = {
        "date": dt.date.today().isoformat(),
        "coach_style": coach_style,
        "completion_rate": completion_rate,
        "completed_count": completed_count,
        "total_habits": total_habits,
        "mood": mood,
        "mood_label": mood_label,
        "city": city,
        "weather": weather,
        "dog_breed": dog_breed,
        "report": report,
        "habits_checked": {k: v for k, v in habits_checked.items()},
    }

    share_text = (
        "📊 AI 습관 트래커 공유\n"
        f"- 날짜: {share_payload['date']}\n"
        f"- 코치: {coach_style}\n"
        f"- 달성률: {completion_rate}% ({completed_count}/{total_habits})\n"
        f"- 기분: {mood}/10 ({mood_label})\n"
        f"- 도시: {city}\n"
        f"- 날씨: {weather.get('desc') if weather else '없음'} / {weather.get('temp_c') if weather else '-'}°C\n"
        f"- 강아지: {dog_breed or '없음'}\n\n"
        "✅ 체크한 습관\n"
        + "\n".join([f"- {k}" for k, v in habits_checked.items() if v])
        + ("\n(없음)\n" if completed_count == 0 else "\n")
        + "\n📝 리포트\n"
        f"{report or '생성 실패'}\n"
    )

    st.markdown("### 🔗 공유용 텍스트")
    st.code(share_text, language="text")

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
