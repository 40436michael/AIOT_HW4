import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

# ======================
# Streamlit UI 設定
# ======================
st.set_page_config(
    page_title="AIGC Image Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 AIGC 圖像生成 Demo")
st.write("基於 Stable Diffusion 的文字生成圖像範例（CPU 版）")

# ======================
# 載入模型（只載一次）
# ======================
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")
    return pipe

pipe = load_model()

# ======================
# 使用者輸入
# ======================
prompt = st.text_input(
    "輸入文字描述（Prompt）",
    value="A cute cat, digital art"
)

steps = st.slider(
    "Inference Steps",
    min_value=10,
    max_value=50,
    value=25
)

generate_btn = st.button("生成圖片")

# ======================
# 產生圖片
# ======================
if generate_btn:
    with st.spinner("圖片生成中，CPU 模式請稍等..."):
        image = pipe(
            prompt,
            num_inference_steps=steps
        ).images[0]

    st.image(image, caption="Generated Image", use_column_width=True)
