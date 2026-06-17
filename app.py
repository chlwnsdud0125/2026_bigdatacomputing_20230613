# Streamlit 대시보드 (app.py)
import streamlit as st # 웹 화면을 만드는 핵심 라이브러리
import numpy as np # 숫자 계산 도구
import joblib # 학습된 모델 파일(.pkl)을 불러오는 도구
import pandas as pd # 표(테이블)를 다루는 도구
import matplotlib.pyplot as plt # 그래프 그리는 도구

# 웹 화면 설정을 와이드하게 설정
st.set_page_config(page_title="기대수명 예측 파이프라인", layout="wide")

# 대시보드 제목과 설명 문구 출력
st.title("🌍 WHO 기대수명 다중 회귀 예측 대시보드")
st.write("선형 회귀, 다항 회귀, 릿지(Ridge) 규제 모델의 성능(과대적합)을 비교하고, 특성값 조절을 통해 기대수명을 실시간으로 예측합니다.")
st.markdown("---") # 가로 구분선

# 1. 모델 파일 로드
def load_model_data():
    try:
        return joblib.load("life_expectancy_models.pkl") # 저장된 모델 파일 불러오기
    except FileNotFoundError:
        st.error("⚠️ 'life_expectancy_models.pkl' 파일이 없습니다.") # 파일 없으면 에러 메시지
        return None

payload = load_model_data() # 데이터 불러오기 실행

if payload is not None: # 파일이 정상적으로 로드됐을 때 아래 기능 수행
    models = payload["models"]
    cv_results = payload["cv_results"]
    feature_names = payload["features"]

    # 모델 성능 평가 표 및 시각화 영역
    st.subheader("📊 3종 회귀 파이프라인 성능 비교 (50개 Train 샘플 기준)")

    col1, col2 = st.columns((1, 1)) # 화면을 좌우 2칸으로 나눔

    with col1: # 왼쪽 칸
        st.markdown("**1. 모델 성능 평가지표 테이블**")
        summary_data = []
        for model_name, info in cv_results.items(): # 각 모델의 성적을 하나씩 꺼냄
            summary_data.append({
                "모델명": model_name,
                "Complexity (특성 수)": info["complexity"],
                "Train R² 점수": round(info["train_r2"], 4),
                "Test R² 점수": round(info["test_r2"], 4),
                "Train MSE": round(info["train_mse"], 2),
                "Test MSE": round(info["test_mse"], 2)
            })
        df_summary = pd.DataFrame(summary_data) # 표 데이터로 변환
        st.dataframe(df_summary, use_container_width=True, hide_index=True) # 웹에 표 출력

        st.caption("ℹ️ Poly 모델의 과대적합(Overfitting) 확인 가능")

    with col2: # 오른쪽 칸
        st.markdown("**2. Test R² (결정계수) 비교 막대그래프**")
        fig, ax = plt.subplots(figsize=(6, 4)) # 그래프 그릴 준비

        names = list(cv_results.keys())
        test_r2_scores = [info["test_r2"] for info in cv_results.values()]

        colors = ['skyblue', 'salmon', 'lightgreen'] # 막대 색상 설정
        bars = ax.bar(names, test_r2_scores, color=colors) # 막대 그래프 생성

        ax.set_ylabel('Test R² Score')

        # 막대 위에 숫자 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom')

        st.pyplot(fig) # 그래프를 웹 화면에 출력

    st.markdown("---")

    # 사이드바 입력 및 실시간 동적 예측 UI
    st.sidebar.header("📋 새로운 데이터 입력 (Features)")

    # 사용자가 직접 조절하는 슬라이더 생성
    input_am = st.sidebar.slider("Adult Mortality (성인 사망률)", min_value=1, max_value=750, value=150)
    input_bmi = st.sidebar.slider("BMI (체질량 지수)", min_value=1.0, max_value=90.0, value=38.0, step=0.1)
    input_gdp = st.sidebar.slider("GDP (1인당 국내총생산)", min_value=1.0, max_value=120000.0, value=5000.0, step=100.0)

    st.sidebar.markdown("---")

    # 모델 선택용 드롭다운 박스
    st.sidebar.header("⚙️ 모델 선택")
    selected_model_name = st.sidebar.selectbox(
        "예측에 사용할 파이프라인 모델을 선택하세요:",
        ("Linear", "Poly", "Ridge")
    )

    # 입력값을 모델이 이해할 수 있는 배열로 변환
    newdata = np.array([[input_am, input_bmi, input_gdp]])

    # 선택된 모델 꺼내오기
    selected_pipeline = models[selected_model_name]

    # 실시간 예측 수행 (학습된 모델이 정답을 추측)
    predicted_life = selected_pipeline.predict(newdata)[0]

    # 결과 화면 출력
    st.subheader(f"🔮 실시간 기대수명 예측 결과 [{selected_model_name} 모델]")

    # 예측값 강조 표시
    st.metric(
        label="예측된 기대수명 (Life Expectancy)",
        value=f"{predicted_life:.1f} 세",
        delta=f"선택 모델: {selected_model_name}"
    )

    st.info(f"📌 **현재 입력값** 👉 Adult Mortality: {input_am} | BMI: {input_bmi} | GDP: {input_gdp}")
