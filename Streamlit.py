# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import json
import warnings


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
sys.path.append(os.path.dirname(resource_path('NN_numpy.py')))
import NN_numpy

warnings.filterwarnings("ignore", category=UserWarning, message="iCCP: known incorrect sRGB profile")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- Streamlit 界面 ---
st.set_page_config(page_title="混凝土强度预测", layout="wide")
st.title("混凝土强度预测程序 📈")
if 'results' not in st.session_state:
    st.session_state.results = None
tab1, tab2 = st.tabs(["📁 文件预测", "✍️ 手动输入"])

# --- 文件预测标签页 ---
with tab1:
    st.header("通过上传 .csv 文件进行预测")
    uploaded_file = st.file_uploader("选择一个 CSV 文件", type="csv")

    if st.button("开始预测 (文件)", key="predict_file"):
        filepath = None
        if uploaded_file is not None:
            filepath = uploaded_file
        else:
            default_path = resource_path("concrete_test.csv")
            if os.path.exists(default_path):
                filepath = default_path
                st.info(f"未上传文件，已自动加载默认测试文件: `concrete_test.csv`")
            else:
                st.error("请上传一个文件，或确保 `concrete_test.csv` 在项目目录中。")

        if filepath is not None:
            try:
                results = NN_numpy.data_test(filepath)
                st.session_state.results = results
                st.success("文件预测成功！结果已在下方显示。")
            except Exception as e:
                st.error(f"处理文件时发生错误: {e}")

# --- 手动输入标签页 ---
with tab2:
    st.header("手动输入数据进行预测")
    st.caption(
        "每行输入8个或9个由逗号或空格分隔的数值。顺序: Cement, Blast_Furnace_Slag, Fly_ash, Water, Superplasticizer, Coarse_aggregate, Fine_aggregate, Age, (可选: Concrete_compressive_strength)")

    manual_input_text = st.text_area(
        "在此处输入数据:",
        height=200,
        placeholder="例如:\n264.0 0.0 111.0 180.0 9.0 932.0 670.0 28 35.2\n540.0 0.0 0.0 162.0 2.5 1040.0 676.0 28"
    )

    if st.button("开始预测 (手动)", key="predict_manual"):
        raw_input = manual_input_text.strip()

        # 检查彩蛋
        if raw_input == '戴松芸':
            st.session_state.results = "easter_egg"
            st.balloons()
            st.info("触发彩蛋！请查看下方结果区域。")
        else:
            lines = [line for line in raw_input.split('\n') if line.strip()]
            if not lines:
                st.warning("请输入至少一行数据。")
            else:
                all_features, true_values = [], []
                has_true_values = any(len(line.replace(',', ' ').split()) == 9 for line in lines)
                try:
                    for i, line in enumerate(lines):
                        parts = line.replace(',', ' ').split()
                        numbers = [float(p) for p in parts]
                        if len(numbers) == 8:
                            all_features.append(numbers)
                            if has_true_values: true_values.append(None)
                        elif len(numbers) == 9:
                            all_features.append(numbers[:8])
                            true_values.append(numbers[8])
                        else:
                            raise ValueError(f"第 {i + 1} 行输入了 {len(numbers)} 个数字，需要8或9个。")

                    predictions = NN_numpy.model.predict(np.array(all_features))
                    st.session_state.results = {
                        'predictions': predictions,
                        'true_values': true_values if has_true_values else None
                    }
                    st.success("手动输入预测成功！结果已在下方显示。")
                except Exception as e:
                    st.error(f"处理输入数据时发生错误: {e}")

st.divider()

# --- 结果展示区域 ---
st.header("预测结果")

if st.session_state.results is None:
    st.info("请先在上方进行预测，结果将在此处显示。")
elif st.session_state.results == "easter_egg":
    st.subheader("老弟，压力！")
    try:
        json_path = resource_path("easter_egg_image.json")
        with open(json_path, 'r') as f:
            image_list = json.load(f)
        img_array = np.array(image_list, dtype=np.uint8)

        fig, ax = plt.subplots()
        ax.imshow(img_array)
        ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"加载彩蛋时出错: {e}")

# 处理正常的预测结果
else:
    results = st.session_state.results
    predictions = results['predictions']
    true_values = results['true_values']
    data_for_table = {"行号": range(1, len(predictions) + 1), "预测值": predictions}
    if true_values is not None:
        data_for_table["真实值"] = true_values

    df = pd.DataFrame(data_for_table)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("数据详情")
        st.dataframe(df.style.format({"预测值": "{:.2f}", "真实值": "{:.2f}"}), width='stretch')

    with col2:
        st.subheader("性能评估")
        fig, ax = plt.subplots()

        if true_values is not None:
            valid_preds = np.array([p for p, t in zip(predictions, true_values) if t is not None])
            valid_trues = np.array([t for t in true_values if t is not None])

            if len(valid_trues) > 0:
                ax.scatter(valid_trues, valid_preds, alpha=0.7, label="数据点")
                lims = [min(valid_trues.min(), valid_preds.min()), max(valid_trues.max(), valid_preds.max())]
                ax.plot(lims, lims, 'r--', alpha=0.75, label="理想情况 (y=x)")
                ax.set_title("预测值 vs. 真实值")
                ax.set_xlabel("真实值")
                ax.set_ylabel("预测值")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)

                mse = np.mean((valid_preds - valid_trues) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(valid_preds - valid_trues))
                st.metric(label="均方根误差 (RMSE)", value=f"{rmse:.4f}")
                st.metric(label="平均绝对误差 (MAE)", value=f"{mae:.4f}")

                if len(valid_trues) > 1:
                    ss_res = np.sum((valid_trues - valid_preds) ** 2)
                    ss_tot = np.sum((valid_trues - np.mean(valid_trues)) ** 2)
                    if ss_tot > 0:
                        r2 = 1 - (ss_res / ss_tot)
                        st.metric(label="R² 分数", value=f"{r2:.4f}")
            else:
                st.info("没有有效的真实值用于计算指标和绘图。")
        else:
            st.info("未提供真实值，无法进行性能评估和绘图。")
