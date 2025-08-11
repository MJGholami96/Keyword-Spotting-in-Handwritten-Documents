# 📝 Persian Keyword Spotting — "تاریخ" in Handwritten Forms

Detect the printed Persian keyword **"تاریخ"** ("date") in scanned handwritten forms using **Mask R-CNN** with custom anchors and data augmentation.

<p align="center">
  <img src="docs/sample_output.jpg" alt="Sample detection" width="500">
</p>

---

## 📌 Overview
This project implements a deep learning pipeline to **detect the printed keyword "تاریخ"** in Persian (Farsi) handwritten forms.  
The system is designed for automating document processing tasks such as:
- Locating handwritten dates next to the keyword
- Enabling fast document indexing
- Supporting downstream applications in information retrieval

Our approach uses **Mask R-CNN** with:
- **Transfer learning** from MS COCO dataset
- **Custom anchor optimization** for Persian word shapes
- **Data augmentation** to boost recall and robustness

---

## ✨ Features
- Detects the **printed keyword "تاریخ"** in scanned forms
- Optimized for Persian fonts and form layouts
- **Custom anchor sizes** for higher precision
- **Augmentation pipeline** for better generalization
- Works with **TIFF, JPG, PNG** images
- Includes training & inference scripts

---

## 📁 Project Structure
