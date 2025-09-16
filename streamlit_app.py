import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime

# --- 기본 설정 ---
st.set_page_config(page_title="기후 변화와 학업 성취 대시보드", layout="wide")

# Pretendard 폰트 설정
FONT_PATH = "/fonts/Pretendard-Bold.ttf"
if os.path.exists(FONT_PATH):
    FONT_NAME = "Pretendard"
else:
    FONT_NAME = None  # 폰트 없으면 기본 폰트 사용

def apply_plotly_font(fig):
    if FONT_NAME:
        fig.update_layout(font=dict(family=FONT_NAME))
    return fig

# --- 데이터 로딩 함수 ---
@st.cache_data
def load_noaa_data():
    """
    NOAA: Global Ocean Surface Temperature Anomaly Dataset
    출처: https://psl.noaa.gov/data/timeseries/
    """
    url = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-monthly/access/anomalies.csv"
    try:
        df = pd.read_csv(url)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[df["date"] <= pd.Timestamp.today()]  # 미래 데이터 제거
        df = df.rename(columns={"anomaly": "value"})
        df = df[["date", "value"]]
        return df, True
    except Exception:
        # fallback 예시 데이터
        dates = pd.date_range("2000-01-01", periods=240, freq="M")
        values = np.sin(np.linspace(0, 20, 240)) + np.random.normal(0, 0.2, 240)
        df = pd.DataFrame({"date": dates, "value": values})
        return df, False

@st.cache_data
def load_user_data():
    """
    사용자 입력 기반 예시 데이터 (보고서 설명 반영)
    기온 상승 vs 수학 점수
    """
    np.random.seed(42)
    years = np.arange(2000, 2021)
    temps = 22 + 0.05 * (years - 2000) + np.random.normal(0, 0.3, len(years))
    scores = 500 - (temps - 22) * 5 + np.random.normal(0, 5, len(years))
    df = pd.DataFrame({
        "date": pd.to_datetime(years, format="%Y"),
        "summer_avg_temp_C": temps,
        "math_score": scores
    })
    return df

# --- 대시보드 제목 ---
st.title("🌡️ 기후 변화와 학업 성취 대시보드")

# --- (1) 공개 데이터 대시보드 ---
st.header("① NOAA 공개 데이터: 전 지구 해수면 온도 이상치")
noaa_df, success = load_noaa_data()
if not success:
    st.warning("⚠️ NOAA 데이터 API 호출 실패 → 예시 데이터로 대체했습니다.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("시계열 추세")
    fig = px.line(noaa_df, x="date", y="value", labels={"value": "온도 이상치 (°C)", "date": "날짜"})
    fig = apply_plotly_font(fig)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("이동평균 (12개월)")
    df_ma = noaa_df.copy()
    df_ma["MA12"] = df_ma["value"].rolling(12).mean()
    fig2 = px.line(df_ma, x="date", y="MA12", labels={"MA12": "12개월 이동평균 (°C)", "date": "날짜"})
    fig2 = apply_plotly_font(fig2)
    st.plotly_chart(fig2, use_container_width=True)

st.caption("📊 NOAA 데이터 출처: [GHCN Monthly Anomalies](https://www.ncei.noaa.gov/data/global-historical-climatology-network-monthly/access/anomalies.csv)")

csv = noaa_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 NOAA 데이터 다운로드", csv, "noaa_data.csv", "text/csv")

# --- (2) 사용자 입력 대시보드 ---
st.header("② 사용자 연구 데이터: 기온과 학업 성취")

user_df = load_user_data()

# 사이드바 옵션
st.sidebar.header("데이터 옵션")
min_date = user_df["date"].min().date()
max_date = user_df["date"].max().date()

date_range = st.sidebar.date_input(
    "기간 선택",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

smoothing_window = st.sidebar.slider("이동평균 윈도우(데이터 포인트 수)", 0, 5, 0, help="0이면 스무딩 미적용")
standardize = st.sidebar.checkbox("수학 점수 표준화(Z-score) 표시", value=False)

# 필터링
df_vis = user_df[(user_df["date"] >= start_date) & (user_df["date"] <= end_date)].copy()
if smoothing_window > 0:
    df_vis["math_score"] = df_vis["math_score"].rolling(smoothing_window).mean()
if standardize:
    df_vis["math_score"] = (df_vis["math_score"] - df_vis["math_score"].mean()) / df_vis["math_score"].std()

# 시각화 1: 시계열
st.subheader("연도별 수학 점수 및 여름 평균 기온")
fig3 = px.line(df_vis, x="date", y=["summer_avg_temp_C", "math_score"],
               labels={"value": "값", "date": "연도", "variable": "지표"},
               title="연도별 기온과 수학 점수")
fig3 = apply_plotly_font(fig3)
st.plotly_chart(fig3, use_container_width=True)

# 시각화 2: 산점도
st.subheader("기온 vs 수학 점수 (산점도)")
try:
    scatter_trend = px.scatter(
        df_vis,
        x="summer_avg_temp_C",
        y="math_score",
        trendline="ols",
        labels={"summer_avg_temp_C": "여름 평균기온 (°C)", "math_score": "수학 점수"},
        title="여름 평균기온 vs 수학 점수 (회귀선 포함: OLS)"
    )
    scatter_trend = apply_plotly_font(scatter_trend)
    st.plotly_chart(scatter_trend, use_container_width=True)
except Exception:
    st.error("산점도 시각화 중 오류 발생. numpy polyfit으로 대체합니다.")
    coeffs = np.polyfit(df_vis["summer_avg_temp_C"], df_vis["math_score"], 1)
    poly_line = np.polyval(coeffs, df_vis["summer_avg_temp_C"])
    fig4 = px.scatter(df_vis, x="summer_avg_temp_C", y="math_score")
    fig4.add_traces(px.line(x=df_vis["summer_avg_temp_C"], y=poly_line).data)
    fig4 = apply_plotly_font(fig4)
    st.plotly_chart(fig4, use_container_width=True)

# 기초 통계량 + 상관계수
st.subheader("📈 기초 통계 및 상관관계")
st.write("데이터 기본 요약 (평균, 표준편차 등):")
st.dataframe(df_vis[["summer_avg_temp_C", "math_score"]].describe().T)

corr_value = df_vis["summer_avg_temp_C"].corr(df_vis["math_score"])
st.metric("여름 평균기온과 수학 점수 상관계수 (Pearson)", f"{corr_value:.3f}")

csv2 = df_vis.to_csv(index=False).encode("utf-8")
st.download_button("📥 사용자 데이터 다운로드", csv2, "user_data.csv", "text/csv")

# --- 전체 출처 모음 ---
st.markdown("---")
st.markdown("### 📚 참고 출처")
st.markdown("- NOAA (National Oceanic and Atmospheric Administration), [GHCN Monthly Anomalies](https://www.ncei.noaa.gov/data/global-historical-climatology-network-monthly/access/anomalies.csv)")
st.markdown("- Park, R. J., & Goodman, J. (2023). *Heat and Learning*. PLOS Climate.")
st.markdown("- OECD PISA 데이터 및 관련 학업 성취도 연구 보고서")
st.markdown("- 기상청 기후자료개방포털, 최근 20년간 전국 폭염일수 통계")
