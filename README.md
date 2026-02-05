# Automatic Face Grouping Web App (No Training, No Names)

A Flask-based web application that automatically detects and groups faces from images and videos **without any prior training, manual labeling, or person names**.

This project uses **pre-trained face embeddings** and **unsupervised clustering (DBSCAN)** to identify and organize unique individuals from uploaded media. It is designed to work out-of-the-box for small to medium-sized photo collections such as family albums or company event photos.

---

## 🚀 Features

- Upload a folder of images (`.jpg`, `.png`, `.jpeg`)
- Optional support for video files (`.mp4`, `.avi`)
- Automatic face detection using **dlib / face_recognition**
- Face embedding extraction using pre-trained models
- **Unsupervised face clustering** with DBSCAN
- Automatically determines the number of unique people
- Creates person-wise folders:
```

output/
├── Person_1/
├── Person_2/
├── Person_3/
└── Unknown/

```
- Group photos containing multiple people are copied into each relevant folder
- Faces that do not confidently belong to any cluster are stored in **Unknown**
- Simple Flask web interface (upload → process → results)
- Downloadable organized output directory
- Beginner-friendly, well-commented code

---

## 🧠 How It Works (High-Level Logic)

1. **Upload Media**
 - User uploads a folder of images (and optionally videos) via the Flask UI.

2. **Face Detection**
 - Faces are detected in each image or extracted video frame using `face_recognition`.

3. **Embedding Extraction**
 - Each detected face is converted into a 128-D numerical embedding using a pre-trained dlib model.

4. **Unsupervised Clustering**
 - All face embeddings are clustered using **DBSCAN**, which:
   - Does not require the number of people beforehand
   - Automatically groups similar faces
   - Marks outliers as noise (`Unknown`)

5. **Media Organization**
 - Person-wise folders are created automatically.
 - Images containing multiple faces are copied into multiple folders.
 - Unclustered faces are stored in the `Unknown` folder.

6. **Results**
 - The user can view the generated folders and download the organized output.

---

## 🛠️ Tech Stack

| Component | Technology |
|---------|------------|
| Backend | Python, Flask |
| Face Detection & Embeddings | face_recognition (dlib) |
| Clustering | scikit-learn (DBSCAN) |
| Image / Video Processing | OpenCV |
| File Handling | os, shutil |

---

## 📁 Project Structure

```

auto-face-grouping/
│
├── app.py
│
├── utils/
│   ├── face_utils.py
│   ├── clustering.py
│   ├── video_utils.py
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   └── style.css
│
├── uploads/
│
├── output/
│
├── requirements.txt
│
└── README.md

````

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/auto-face-grouping.git
cd auto-face-grouping
````

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** Installing `dlib` may take time. Make sure CMake and a C++ compiler are installed.

---

## ▶️ Running the Application (VS Code / Terminal)

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 🎥 Video Processing (Optional)

* Videos are **not** processed by extracting frames every **N seconds**
* Just create a folder videos and dump all the videos their
* Output structure:

  ```
  output/
    └── videos/
        ├── video1.mkv
        ├── video2.avi
        └── video3.mp4
  ```

---

## ⚠️ Constraints & Design Decisions

* No deep learning training
* No manual person naming
* Uses only pre-trained models
* Fully automatic workflow
* Optimized for **small to medium datasets**
* Not intended for real-time or large-scale surveillance

---

## ❗ Error Handling

* Images with no detected faces are safely skipped
* Corrupted or unsupported files are ignored
* Faces not confidently clustered are assigned to `Unknown`

---

## 🎯 Use Cases

* Family photo organization
* Event photo sorting
* Small company media archives
* AI/ML portfolio project
* Demonstration of unsupervised learning in computer vision

---

## 📌 Future Improvements

* Add confidence visualization
* Improve UI with progress bar
* Allow ZIP upload/download
* Add face preview per cluster
* GPU acceleration support

---

## 🧑‍💻 Author

**Lovnish Verma**
AI / ML Engineer | Python Developer

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub — it helps others discover it!


---



Just tell me 👍
```
