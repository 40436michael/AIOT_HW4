import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.set_page_config(page_title="Anime Lineart WebUI", page_icon="🎨", layout="wide")

st.title("🎨 Anime Lineart / Manga-like Style (CPU + LoRA)")
st.write("文字生成動漫線稿風格圖像介面")

@st.cache_resource
def load_pipeline():
    torch.set_num_threads(2)
    # 基模型
    pipe = StableDiffusionPipeline.from_pretrained(
        "hakurei/waifu-diffusion",
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")
    # 載入 LoRA
    pipe.load_lora_weights("lora/lineart_manga.safetensors", weight=1.0)
    return pipe

pipe = load_pipeline()

# UI
col1, col2 = st.columns([1, 2])

with col1:
    prompt = st.text_area("Prompt", "cute anime girl, long hair, detailed")
    negative_prompt = st.text_area("Negative Prompt", "blurry, low quality, bad anatomy")
    steps = st.slider("Sampling Steps", 10, 50, 25)
    cfg = st.slider("CFG Scale", 1.0, 15.0, 7.5)
    seed = st.number_input("Seed (-1 為隨機)", min_value=-1, value=-1)
    generate_btn = st.button("生成圖片 🚀")

with col2:
    st.subheader("生成結果")
    if generate_btn:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        generator = None if seed==-1 else torch.Generator("cpu").manual_seed(seed)

        def callback(step, timestep, latents):
            progress = int((step+1)/steps*100)
            progress_text.text(f"生成進度：{progress}% (Step {step+1}/{steps})")
            progress_bar.progress(progress)

        with st.spinner("生成中...（CPU 模式，請耐心等待）"):
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                callback=callback,
                callback_steps=1
            ).images[0]

        progress_text.text("生成完成 ✅")
        progress_bar.progress(100)
        st.image(image, use_container_width=True)
