# streamlit_app.py
"""
Streamlit 앱: "기온 상승과 학업·수면 영향" 대시보드 (에러를 줄이고 안정적으로 동작하도록 재작성)
- 출력: 공개 데이터(참고 페이지 접속 확인 후 예시/모의 데이터 사용) + 사용자 프롬프트 기반 내장 데이터
- 모든 UI는 한국어로 표기
- 오늘(로컬 자정, Asia/Seoul) 이후 날짜는 자동 제거
- 전처리된 표 CSV 다운로드 제공
- 폰트: /fonts/Pretendard-Bold.ttf 적용을 시도 (없으면 무시)
"""

import io
import os
from datetime import datetime
import pytz
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go

# 페이지 설정
st.set_page_config(page_title="기온 상승과 학업·수면 영향 대시보드", layout="wide")
KST = pytz.timezone("Asia/Seoul")
TODAY = datetime.now(KST).date()

# 폰트 적용 시도 (Pretendard)
FONT_PATH = "/fonts/Pretendard-Bold.ttf"
FONT_NAME = None
if os.path.exists(FONT_PATH):
    # Streamlit 환경에서 로컬 ttf를 전역으로 등록하는 작업은 제한적일 수 있음.
    # Plotly에선 레이아웃에서 font.family를 직접 지정해 적용 시도.
    FONT_NAME = "Pretendard"

# 유틸리티: 미래 날짜 제거(로컬 KST 기준)
def remove_future_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df[df[date_col].dt.date <= TODAY].reset_index(drop=True)

# Plotly에 폰트 적용 도우미
def apply_plotly_font(fig):
    if FONT_NAME:
        fig.update_layout(font=dict(family=FONT_NAME))
    return fig

# ---------- 공개 데이터(참고 페이지 연결 + 예시 데이터) ----------
@st.cache_data(ttl=3600)
def fetch_official_sst_timeseries():
    """
    외부(공식) 데이터 접근을 '참조 페이지 확인' 방식으로 시도.
    - 외부 접속이 성공하면 'online_reference' 상태를 반환(실제 원시 데이터는 앱 배포 환경에서 엔드포인트를 지정해 연결 권장).
    - 실패 시 'fallback_example'으로 소규모 모의 시계열 데이터를 생성해 반환.
    반환 형식: {"status": str, "source_url": Optional[str], "data": Optional[pd.DataFrame]}
    """
    reference_urls = [
        "https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html",
        "https://www.ncei.noaa.gov/products/climate-data-records/pathfinder-sea-surface-temperature",
        "https://www.ncei.noaa.gov/products/land-based-station/noaa-global-temp",
    ]
    session = requests.Session()
    session.headers.update({"User-Agent": "streamlit-dashboard/1.0"})
    for url in reference_urls:
        try:
            r = session.get(url, timeout=6)
            if r.status_code == 200:
                return {"status": "online_reference", "source_url": url, "data": None}
        except Exception:
            continue

    # 모두 실패 -> 안정적인 대체(모의) 데이터 생성 (연단위 샘플)
    years = np.arange(1981, datetime.now().year + 1)
    dates = pd.to_datetime([f"{y}-07-01" for y in years])
    # 모의 해수면 온도: 완만한 상승 추세 + 계절성 노이즈
    sst = 16.5 + 0.015 * (years - 1981) + np.sin((years - 1981) / 4.0) * 0.12 + np.random.normal(0, 0.05, len(years))
    df = pd.DataFrame({"date": dates, "sst_C": np.round(sst, 3)})
    df = remove_future_dates(df, "date")
    return {"status": "fallback_example", "source_url": None, "data": df}

# ---------- 사용자 프롬프트 기반 내장 데이터 ----------
@st.cache_data(ttl=3600)
def load_user_prompt_dataset():
    """
    사용자가 제공한 보고서(프롬프트) 내용을 반영해 연단위 예시 데이터 생성.
    표준화 규칙: date, value(다양한 컬럼 명), group(optional)
    반환: DataFrame (date, summer_avg_temp_C, avg_sleep_hours, math_score)
    """
    years = np.arange(2015, datetime.now().year + 1)
    dates = pd.to_datetime([f"{y}-06-30" for y in years])
    # 기온(예시): 2015년 기준에서 점진 상승
    temps = 25.0 + 0.25 * (years - 2015) + np.random.normal(0, 0.45, len(years))
    # 수면시간: 온도 상승시 하락하는 경향
    sleep_hours = 8.0 - 0.06 * (temps - temps.mean()) - 0.02 * (years - 2015) + np.random.normal(0, 0.08, len(years))
    # 수학 점수: 표준화된 점수(예: 평균 500) — 온도 증가시 약간 하락하는 가정
    math_score = 500 - 2.2 * (temps - temps.mean()) - 0.25 * (years - 2015) + np.random.normal(0, 3.5, len(years))

    df = pd.DataFrame({
        "date": dates,
        "summer_avg_temp_C": np.round(temps, 2),
        "avg_sleep_hours": np.round(sleep_hours, 2),
        "math_score": np.round(math_score, 1)
    })
    df = remove_future_dates(df, "date")
    return df

# ---------- 앱 UI ----------
st.title("기온 상승과 학업·수면 영향 — 안정화된 재구현")
st.caption("에러 발생 시 재작성된 앱입니다. (모든 레이블은 한국어)")

# 공개 데이터 섹션
st.header("1) 공식 공개 데이터 (참고 페이지 연결 후 예시 데이터 표시)")
info = fetch_official_sst_timeseries()

if info["status"] == "online_reference":
    st.success("공식 데이터 참조 페이지 접근 성공")
    st.write(f"참조 URL: {info['source_url']}")
    st.info("참고: 앱은 대형 기후 데이터(예: NetCDF)를 배포 환경에서 직접 연결해 처리하는 것을 권장합니다.")
    # 모의 시각화: 간단한 연도 추세(모의 데이터 생성해서 시각화)
    years = np.arange(1981, datetime.now().year + 1)
    dates = pd.to_datetime([f"{y}-07-01" for y in years])
    sst_mock = 16.5 + 0.015 * (years - 1981) + np.sin((years - 1981) / 4.0) * 0.12
    df_mock = pd.DataFrame({"date": dates, "sst_C": np.round(sst_mock, 3)})
    df_mock = remove_future_dates(df_mock, "date")
    fig = px.line(df_mock, x="date", y="sst_C", title="(참조) 전지구 해수면 온도 추세 (모의 시각화)",
                  labels={"date": "연도", "sst_C": "해수면 온도 (°C)"})
    fig = apply_plotly_font(fig)
    st.plotly_chart(fig, use_container_width=True)

    # CSV 다운로드(모의)
    buf = io.StringIO()
    df_mock.to_csv(buf, index=False)
    st.download_button("공개 데이터(모의) CSV 다운로드", buf.getvalue(), file_name="official_sst_mock.csv", mime="text/csv")

else:
    st.warning("공식 데이터 페이지 접근 실패 — 예시(대체) 데이터로 표시합니다.")
    df_official = info["data"]
    st.dataframe(df_official.head(30))
    fig = px.line(df_official, x="date", y="sst_C", title="대체 예시: 연도별 해수면 온도 (예시 데이터)",
                  labels={"date": "연도", "sst_C": "해수면 온도 (°C)"})
    fig = apply_plotly_font(fig)
    st.plotly_chart(fig, use_container_width=True)
    buf = io.StringIO()
    df_official.to_csv(buf, index=False)
    st.download_button("대체 공개 데이터 CSV 다운로드", buf.getvalue(), file_name="official_sst_example.csv", mime="text/csv")

st.markdown("---")

# 사용자 데이터(프롬프트 기반) 섹션
st.header("2) 사용자 입력(프롬프트) 기반 데이터 대시보드 (내장 데이터 사용)")
st.write("프롬프트로 제공한 보고서 내용을 반영한 내장 데이터를 사용합니다. 앱 실행 중 별도 업로드를 요구하지 않습니다.")

user_df = load_user_prompt_dataset()
st.subheader("원본 데이터(내장)")
st.dataframe(user_df)

# 사이드바 옵션 (자동 구성)
st.sidebar.header("데이터 옵션")
min_date = user_df["date"].min().date()
max_date = user_df["date"].max().date()
# 기본값은 전체 기간
date_range = st.sidebar.date_input("기간 선택", value=(min_date, max_date), min_value=min_date, max_value=max_date)
# date_input가 하나의 날짜만 반환할 수 있으므로 안전하게 처리
if isinstance(date_range, tuple) or isinstance(date_range, list):
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date = end_date = pd.to_datetime(date_range)

smoothing_window = st.sidebar.slider("이동평균 윈도우(데이터 포인트 수)", 0, 5, 0, help="0이면 스무딩 미적용")
standardize = st.sidebar.checkbox("수학 점수 표준화(Z-score) 표시", value=False)

# 필터 적용
df_vis = user_df[(user_df["date"] >= start_date) & (user_df["date"] <= end_date)].copy()
if df_vis.empty:
    st.error("선택한 기간의 데이터가 없습니다. 기간 선택을 조정하세요.")
else:
    # 전처리: 결측/중복/형변환 처리
    df_vis = df_vis.drop_duplicates().reset_index(drop=True)
    # 스무딩
    if smoothing_window and smoothing_window > 0:
        # 윈도우가 데이터 포인트 수이므로 index를 활용한 롤링
        df_vis = df_vis.set_index("date").rolling(window=smoothing_window, min_periods=1).mean().reset_index()

    # 시각화 1: 기온 & 수면 시간 추이 (같은 그래프)
    fig1 = px.line(df_vis, x="date", y=["summer_avg_temp_C", "avg_sleep_hours"],
                   labels={"value": "값", "date": "연도", "variable": "지표"},
                   title="여름 평균기온 및 평균 수면시간 추이")
    fig1 = apply_plotly_font(fig1)
    st.plotly_chart(fig1, use_container_width=True)

    # 시각화 2: 수학 점수 추이 (별도)
    fig2 = px.line(df_vis, x="date", y="math_score", labels={"math_score": "수학 평균 점수", "date": "연도"},
                   title="수학 평균 점수 추이")
    fig2 = apply_plotly_font(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    # 산점도: 기온 vs 수학 점수 + 회귀선(가능하면 Plotly trendline 사용, 실패 시 numpy polyfit)
    st.subheader("기온 vs 수학 점수 (산점도)")
    try:
        scatter = px.scatter(df_vis, x="summer_avg_temp_C", y="math_score",
                             labels={"summer_avg_temp_C": "여름 평균기온 (°C)", "math_score": "수학 점수"},
                             title="여름 평균기온 vs 수학 점수 (회귀선 포함: OLS)")
        # Attempt to add trendline using statsmodels via plotly.express (requires statsmodels)
        # If not available or fails, fallback to manual polyfit below.
        scatter_trend = px.scatter(df_vis, x="summer_avg_temp_C", y="math_score", trendline="ols")
        scatter_trend = apply_plotly_font(scatter_trend)
        st.plotly_chart(scatter_trend, use_container_width=True)
    except Exception:
        # fallback: numpy linear fit
        x = df_vis["summer_avg_temp_C"].to_numpy()
        y = df_vis["math_score"].to_numpy()
        if len(x) >= 2:
            coef = np.polyfit(x, y, 1)
            p = np.poly1d(coef)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = p(x_line)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="관측값"))
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="선형회귀(근사)"))
            fig.update_layout(title="여름 평균기온 vs 수학 점수 (근사 회귀선)",
                              xaxis_title="여름 평균기온 (°C)",
                              yaxis_title="수학 점수")
            fig = apply_plotly_font(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("산점도/회귀선을 그리기 위한 충분한 데이터가 없습니다.")

    # 기초 통계 및 상관계수
    st.subheader("기초 통계 및 상관관계")
    st.write(df_vis[["summer_avg_temp_C", "avg_sleep_hours", "math_score"]].describe().round(3))
    corr = df_vis[["summer_avg_temp_C", "avg_sleep_hours", "math_score"]].corr().round(3)
    st.dataframe(corr)

    # 수학 점수 표준화(옵션)
    if standardize:
        mean_ms = df_vis["math_score"].mean()
        std_ms = df_vis["math_score"].std(ddof=0)
        df_vis["math_z"] = ((df_vis["math_score"] - mean_ms) / (std_ms if std_ms != 0 else 1)).round(3)
        fig_z = px.line(df_vis, x="date", y="math_z", labels={"math_z": "수학 점수 (Z-score)", "date": "연도"},
                        title="표준화된 수학 점수(Z-score)")
        fig_z = apply_plotly_font(fig_z)
        st.plotly_chart(fig_z, use_container_width=True)

    # 전처리된 표 CSV 다운로드
    buf = io.StringIO()
    df_vis.to_csv(buf, index=False)
    st.download_button("전처리된 사용자 데이터 CSV 다운로드", buf.getvalue(), file_name="user_prompt_processed.csv", mime="text/csv")

st.markdown("---")
st.markdown("**참고(간단)**: 공개 데이터는 NOAA / KMA의 공식 API 또는 NetCDF 파일을 운영 환경에서 직접 연결해 사용하세요. "
            "앱은 외부 접속 실패 시 자동으로 예시 데이터를 사용하도록 구성되어 있습니다.")
st.markdown("**출처(예시)**")
st.write("- NOAA OISST / Pathfinder / NOAAGlobalTemp (참고 페이지)")
st.write("- 기상청 기후자료포털 (data.kma.go.kr)")

