# Embedded AI Microscopy System for Marine Biodiversity  

> **Automating microscopic biodiversity analysis using Edge AI on Raspberry Pi**  
> A low-cost, offline, and scalable intelligent microscopy solution for marine ecosystem monitoring.

---

## 🌊 Project Overview

Marine biodiversity assessments depend heavily on the microscopic analysis of organisms like **phytoplankton** and **zooplankton**, which are crucial for monitoring ocean health, predicting algal blooms, and managing fisheries.  

However, **manual microscopic analysis** is:

- 🕐 **Time-consuming:** ~20 minutes per sample  
- 👩‍🔬 **Labor-intensive:** Requires skilled taxonomists (often in shortage)  
- ⚖️ **Subjective:** Prone to human error  
- 🚫 **Not scalable:** Unsuitable for large-scale or real-time monitoring
---

## Youtube Video

[![Title](https://img.youtube.com/vi/WEvNaXpfy-I/maxresdefault.jpg)](https://youtu.be/WEvNaXpfy-I)

---

### 🎯 Our Goal  
To develop an **AI-powered embedded microscopy system** that automates detection, classification, and counting of marine microorganisms — directly on low-cost hardware like the **Raspberry Pi 5**, fully **offline** and **field deployable**.

---

## 🚀 Proposed Solution

We designed a **Raspberry Pi–based Embedded Intelligent Microscopy System** that can:

- Capture high-resolution images from a USB digital microscope  
- Run **quantized YOLOv8 Nano models** for real-time detection **and** classification  
- Perform **automated counting and labeling** of detected species  
- Export data in standardized formats (**Darwin Core / OBIS**)  
- Operate **fully offline**, making it ideal for ships and coastal monitoring labs  

---

## ⚙️ System Architecture

```text
Microscope → Image Capture → Detection + Classification (YOLOv8-Nano) → Counting → Dashboard (Streamlit) → Export (JSON/CSV in Darwin Core)
```

---

## 🧩 Hardware Components

| Component        | Specification                     |
| ---------------- | --------------------------------- |
| **Compute Unit** | Raspberry Pi 5 (8GB RAM)          |
| **Microscope**   | USB Digital Microscope (1080p/4K) |
| **Storage**      | 64GB SD Card / SSD                |
| **Power Supply** | 27W PSU or 20,000mAh Power Bank   |
| **Cooling**      | Active Cooling Fan (recommended)  |

---

## 🧰 Software Stack

- **Language:** Python 3.11
- **Frameworks:** PyTorch → ONNX → TensorFlow Lite
- **Model:** YOLOv8 Nano (for detection + classification)
- **Libraries:** OpenCV, NumPy, Streamlit, Matplotlib, Pandas, Ultralytics
- **Interface:** Streamlit web dashboard for live results and performance monitoring

---

## 🧠 AI Models

| Task                           | Model            | Accuracy | Notes                                                  |
| ------------------------------ | ---------------- | -------- | ------------------------------------------------------ |
| **Detection & Classification** | YOLOv8 Nano      | 84%      | Quantized ONNX/TFLite model optimized for Raspberry Pi |
| **Counting**                   | Detection-driven | 76%      | Automatically counts organisms per class               |

### 🧩 Optimizations

- **Post-training quantization** for embedded inference
- **Pruning** for low latency and faster FPS
- **Multi-scale detection** using feature pyramids
- **Image Quality Assessment (IQA)** filters to reject low-quality frames

---

## 🖥️ Dashboard Features

| Feature                                     | Description                                            |
| ------------------------------------------- | ------------------------------------------------------ |
| 🔍 **Real-time Detection & Classification** | Displays bounding boxes, labels, and confidence        |
| 📊 **Performance Metrics**                  | Shows FPS, CPU Usage, Memory Utilization               |
| 🧾 **Automatic Counting**                   | Enumerates detected species per frame                  |
| 💾 **Data Export**                          | Saves outputs in JSON/CSV (Darwin Core / OBIS formats) |
| 🌐 **Hybrid Mode**                          | Optional online model update                           |
| 🧱 **Offline-First Operation**              | Works without internet connectivity                    |

---

## 📅 Prototype Roadmap

| Stage       | Milestone                                          | Status         |
| ----------- | -------------------------------------------------- | -------------- |
| **Stage 1** | Model training (YOLOv8 Nano) & pipeline design     | ✅ Completed    |
| **Stage 2** | Web dashboard integration + laptop demonstration | ✅ Completed|
| **Stage 3** | Full Raspberry Pi integration + optimization       | 🔜 In Progress    |

---

## 🌍 Impact & Applications

| Sector                       | Use Case                                      |
| ---------------------------- | --------------------------------------------- |
| **Marine Research**          | Automated biodiversity monitoring             |
| **Aquaculture**              | Water plankton health & fish safety           |
| **Environmental Monitoring** | Early detection of harmful algal blooms       |
| **Academia**                 | AI microscopy toolkit for teaching & research |
| **Healthcare (Future)**      | Extendable to microbial diagnostics           |

---

## 🇮🇳 Contribution to India & SDGs

- **Make in India:** Indigenous AI hardware + software
- **Digital India:** Digitized biodiversity data collection
- **SDG 14 – Life Below Water:** Marine ecosystem protection
- **SDG 6 – Clean Water:** Freshwater quality monitoring
- **Skill Development:** Promotes embedded AI & research in Indian institutions

---

## 🧩 Expected Outcomes

- ⚙️ Working embedded AI microscopy prototype
- 🔍 Real-time detection, classification, and counting demonstration
- 🧾 Reproducible and documented AI pipeline
- 🌎 Scalable framework for marine research institutions

---

## 🧪 Technical Highlights

- Edge AI inference on **Raspberry Pi 5**
- Quantized **YOLOv8 Nano** model for both detection & classification
- Integrated **IQA filters** for image quality validation
- Federated learning–ready data pipeline
- Total system cost under **₹12,000**
- Power-efficient (10–12W) and **portable** for field use

---

## 👥 Team CodeFather

| Name           | GitHub                                     |
| -------------- | ------------------------------------------ |
| 🧑‍💻 Member 1 | [Madhavan](https://github.com/ssmadhavan006) |
| 🧑‍💻 Member 2 | [Akashgautham](https://github.com/Akashgautham) |
| 🧑‍💻 Member 3 | [Vijaya Karthick](https://github.com/KARTHICK-3056) |
| 🧑‍💻 Member 4 | [Rakshithasri](https://github.com/rakshithasri-k) |
| 🧑‍💻 Member 5 | [Raksha](https://github.com/raksha006) |
| 🧑‍💻 Member 6 | [Divyesh Hari](https://github.com/DIVYESH-HARI) |

> *Team CodeFather – Innovating for Sustainable Marine Ecosystems 🌊*

---

## 🧭 How to Run (Prototype Demo)

```bash
# Clone repository
git clone https://github.com/TeamCodeFather/embedded-ai-microscopy-system.git
cd embedded-ai-microscopy-system

# Install dependencies
pip install -r requirements.txt

# Run dashboard
python app.py
```

> Ensure the **USB digital microscope** is connected and accessible.  
> Raspberry Pi build instructions available in `/docs/pi_setup.md`.

---

## 📜 License

This project is released under the **MIT License**.  
You are free to fork, modify, and improve for educational or research purposes.

---

## 🏁 Final Summary

We are building an **AI-powered, offline, embedded microscopy system** that automates the detection, classification, and counting of marine microorganisms using **YOLOv8 Nano** on the **Raspberry Pi 5**.

This indigenous innovation is **low-cost**, **scalable**, and **field-deployable**, empowering Indian research institutions with advanced biodiversity monitoring tools.

> "From ocean plankton to planetary health — AI at the edge for a sustainable future." 🌎

---

<p align="center">
  <img src="https://img.shields.io/badge/Raspberry%20Pi%205-Edge%20AI-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/YOLOv8-Nano-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Make%20in%20India-🇮🇳-brightgreen?style=for-the-badge"/>
</p>
