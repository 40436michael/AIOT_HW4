# AIGC 圖像生成 Web Demo（Stable Diffusion）

本專案為「物聯網數據分析與應用 — Deep Learning AIGC（HW4）」之實作成果，
參考 Taica AIGC 課程內容，延伸課堂中 Stable Diffusion 之示範，
實作一個文字生成圖像（Text-to-Image）的生成式人工智慧應用，
並以 Streamlit 建立 Web 操作介面。

---

## 專案說明

本系統使用深度學習中的擴散模型（Diffusion Model），
透過 Stable Diffusion 將使用者輸入的文字描述（Prompt）
轉換為對應的圖像內容，展示生成式 AI 在圖像生成上的實際應用。

為降低部署門檻與避免雲端 GPU 限制，
本專案採用 CPU 模式進行推論，重點在於
示範 AIGC 模型流程與 Web 應用整合方式。

---

## 使用技術

- Python
- PyTorch
- Stable Diffusion（Diffusers 套件）
- Streamlit（Web UI）

---

## 執行環境需求

- Python 3.10 以上
- 不需 GPU（CPU 模式）

---

## 安裝套件

```bash
pip install -r requirements.txt
