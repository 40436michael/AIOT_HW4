import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

# ======================
# 頁面設定
# ======================
st.set_page_config(
    page_title="Anime Lineart WebUI",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Anime Lineart / Manga-like Style (CPU + LoRA)")
st.write("文字生成動漫線稿風格圖像介面（CPU 模式）")

# ======================
# 載入模型（只載一次）
# ======================
@st.cache_resource
def load_pipeline():
    # 限制 CPU 線程，避免 Cloud 卡死
    torch.set_num_threads(2)
    
    # 輕量動漫基模
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")
    
    # 載入 LoRA
    pipe.load_lora_weights(
        "lora/animeLineartMangaLike_v30MangaLike.safetensors", 
        weight=0.8
    )
    
    # 確認 LoRA 參數
    for name, param in pipe.unet.named_parameters():
        if "lora" in name:
            print(name, param.shape)

    return pipe

pipe = load_pipeline()

# ======================
# UI 區塊（左設定 / 右結果）
# ======================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("生成參數設定")
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

        # 設定隨機生成器
        generator = None if seed == -1 else torch.Generator("cpu").manual_seed(seed)

        # callback 更新進度條
        def callback(step, timestep, latents):
            progress = int((step + 1) / steps * 100)
            progress = max(0, min(progress, 100))  # 限制 0~100
            progress_text.text(f"生成進度：{progress}% (Step {step+1}/{steps})")
            progress_bar.progress(progress)

        with st.spinner("生成中...（CPU 模式，請耐心等待）"):
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                callback=callback,
                callback_steps=1
            )
            image = output.images[0]

        progress_text.text("生成完成 ✅")
        progress_bar.progress(100)
        st.image(image, caption="Generated Image", use_container_width=True)












