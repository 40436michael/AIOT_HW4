import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

# ======================
# 頁面設定
# ======================
st.set_page_config(
    page_title="Stable Diffusion WebUI (Streamlit)",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Stable Diffusion WebUI（Streamlit）")
st.write("模擬 Stable Diffusion WebUI 的文字生成圖像介面（CPU 版）")

# ======================
# 載入模型（只載一次）
# ======================
@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")
    return pipe

pipe = load_pipeline()

# ======================
# UI 區塊（左設定 / 右結果）
# ======================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("生成參數設定")

    prompt = st.text_area(
        "Prompt",
        "A cute cat, digital art, high quality"
    )

    negative_prompt = st.text_area(
        "Negative Prompt",
        "blurry, low resolution, bad anatomy"
    )

    steps = st.slider("Sampling Steps", 10, 50, 25)
    cfg = st.slider("CFG Scale", 1.0, 15.0, 7.5)

    seed = st.number_input(
        "Seed（-1 為隨機）",
        min_value=-1,
        value=-1
    )

    generate_btn = st.button("生成圖片 🚀")

with col2:
    st.subheader("生成結果")

    if generate_btn:
        with st.spinner("圖片生成中（CPU 模式，請稍候）..."):

            if seed == -1:
                generator = None
            else:
                generator = torch.Generator("cpu").manual_seed(seed)

            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator
            ).images[0]

        st.image(image, caption="Generated Image", use_container_width=True)
