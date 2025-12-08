# AIGC 圖像生成 Web Demo（Stable Diffusion）

本專案為「物聯網數據分析與應用 — Deep Learning AIGC（HW4）」之實作成果，參考 **Taica AIGC** 課程內容，延伸課堂中 Stable Diffusion 的示範，實作一個文字生成圖像（Text-to-Image）的生成式人工智慧應用，並以 **Streamlit** 建立 Web 操作介面。

[https://aiothw4-thjthkhqo5rgsyvucb3jqp.streamlit.app/](https://aiothw4-thjthkhqo5rgsyvucb3jqp.streamlit.app/)

---
## 本地成果
<img width="1897" height="951" alt="image" src="https://github.com/user-attachments/assets/686ec071-a8b3-4d57-8c86-80c257f52503" />
<img width="512" height="512" alt="00000-2084475014" src="https://github.com/user-attachments/assets/d63bbae9-b406-426c-bdfa-8fcf318a2210" />
---
---
## Streamlist APP
<img width="591" height="612" alt="image" src="https://github.com/user-attachments/assets/ea4e6c64-9832-44ad-918e-f95fc790db5f" />

<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/95d151cd-5206-4be0-b1da-8f53e2c8dbc2" />

---
## 專案說明

本系統使用深度學習中的 **擴散模型（Diffusion Model）**，透過 **Stable Diffusion** 將使用者輸入的文字描述（Prompt）轉換為對應的圖像內容，展示生成式 AI 在圖像生成上的實際應用。

為降低部署門檻與避免雲端 GPU 限制，本專案採用 **CPU 模式** 進行推論，重點在於示範 **AIGC 模型流程與 Web 應用整合方式**。

---

## 使用技術

- **Python**  
- **PyTorch**  
- **Stable Diffusion**（Diffusers 套件）  
- **Streamlit**（Web UI）  

---

## 執行環境需求

- Python 3.10 以上  
- 不需 GPU（CPU 模式）  

---

## 安裝套件

可透過 `requirements.txt` 安裝：

```bash
pip install -r requirements.txt
````

必要套件包括：

```bash
pip install torch diffusers transformers streamlit safetensors
```

---

## 使用方式

1. 下載 LoRA 權重檔案：

* [animeLineartMangaLike_v30MangaLike](https://huggingface.co/Sasulee/animeLineartMangaLike_v30MangaLike/tree/main)
  並放置於專案資料夾 `lora/` 內

2. 啟動 Web UI：

```bash
streamlit run app.py
```

3. 開啟瀏覽器，預設網址：

```
http://localhost:8501
```

4. 設定生成參數：

* Prompt / Negative Prompt
* Sampling Steps
* CFG Scale
* Seed (-1 為隨機)

5. 點擊 **生成圖片 🚀**，稍候等待 CPU 生成完成

6. 預覽生成結果，並可下載圖片

---

## 注意事項

* CPU 模式生成速度較慢，建議步數 **20~50**
* 可透過限制 CPU 線程數量提升穩定性：

```python
torch.set_num_threads(2)
```

* 支援 LoRA 權重調整權重，預設 `0.8`
* 可替換不同 LoRA 權重以獲得不同風格效果

---

## 參考資料

* [AI Demo by Yenlung](https://github.com/yenlung/AI-Demo)
* [animeLineartMangaLike_v30MangaLike LoRA on HuggingFace](https://huggingface.co/Sasulee/animeLineartMangaLike_v30MangaLike/tree/main)
* [Diffusers 官方文件](https://huggingface.co/docs/diffusers/index)
* [Stable Diffusion Pipeline 教學](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion)

---

## 版權與授權

* 本專案僅供 **學習與研究使用**
* 模型與 LoRA 權重請遵守原作者授權條款

---

## 致謝

感謝所有提供 **LoRA 權重** 與 **Diffusers 教學** 的開源社群，讓 AI 生成藝術更容易上手！

```






