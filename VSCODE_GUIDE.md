# 🚀 Quick Start Guide - VS Code

## Step-by-Step Instructions to Run in VS Code

### 1. Open Project in VS Code

```bash
# Open VS Code
code .
```

Or:
- Open VS Code → File → Open Folder → Select project folder

---

### 2. Open Terminal in VS Code

- Press `` Ctrl + ` `` (backtick) or
- Menu: Terminal → New Terminal

---

### 3. Create Virtual Environment

**In VS Code Terminal:**

```bash
# Windows
python -m venv venv

# macOS/Linux
python3 -m venv venv
```

---

### 4. Activate Virtual Environment

**Windows (PowerShell):**
```bash
venv\Scripts\activate
```

**Windows (Command Prompt):**
```bash
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Note**: You should see `(venv)` prefix in your terminal

If PowerShell gives execution policy error:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

⏰ **This will take 5-15 minutes** - dlib needs to compile from source.

You'll see a lot of output. Don't worry, it's normal!

---

### 6. Verify Installation

```bash
python -c "import face_recognition; print('Success!')"
```

If you see "Success!" you're ready to go!

---

### 7. Run the Application

```bash
python app.py
```

You should see:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

---

### 8. Open in Browser

Click the link in terminal (Ctrl+Click) or manually navigate to:

**http://localhost:5000**

---

## 🎯 VS Code Extensions (Recommended)

Install these for better development experience:

1. **Python** (Microsoft) - Python language support
2. **Pylance** (Microsoft) - Fast Python language server
3. **Flask Snippets** - Flask code snippets
4. **HTML CSS Support** - HTML/CSS editing

---

## 🐛 Common VS Code Issues

### Issue 1: "Python interpreter not found"

**Solution:**
- Press `Ctrl+Shift+P`
- Type "Python: Select Interpreter"
- Choose the one with `(venv)` in the path

---

### Issue 2: Import errors / Red squiggly lines

**Solution:**
- Make sure virtual environment is activated
- Restart VS Code
- Select correct Python interpreter (see Issue 1)

---

### Issue 3: Terminal shows wrong Python version

**Solution:**
- Make sure `(venv)` is shown in terminal
- If not, reactivate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)

---

## 🔥 Pro Tips for VS Code

### 1. Integrated Git
- View → Source Control (Ctrl+Shift+G)
- Track changes and commit easily

### 2. Debugging
- Set breakpoints by clicking left of line numbers
- Press F5 to start debugging
- Select "Python" → "Flask"

### 3. Multiple Terminals
- Click `+` icon in terminal panel
- Run app in one, test commands in another

### 4. Auto-Save
- File → Auto Save
- Never lose code changes!

### 5. Format Code
- Install "Black Formatter" extension
- Right-click → Format Document
- Or set format on save

---

## 📁 File Explorer in VS Code

```
📂 face-grouping-app
├── 📄 app.py                  ← Main Flask app
├── 📄 face_utils.py          ← Face detection
├── 📄 clustering.py          ← Clustering logic
├── 📄 requirements.txt       ← Dependencies
├── 📄 README.md              ← Documentation
├── 📄 VSCODE_GUIDE.md        ← This file
├── 📂 templates
│   ├── 📄 index.html         ← Upload page
│   └── 📄 results.html       ← Results page
├── 📂 venv                    ← Virtual environment (ignored by git)
├── 📂 uploads                 ← Temporary uploads
└── 📂 output                  ← Organized output
```

---

## ⌨️ Useful VS Code Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + ~` | Open/close terminal |
| `Ctrl + B` | Toggle sidebar |
| `Ctrl + P` | Quick file open |
| `Ctrl + Shift + P` | Command palette |
| `Ctrl + /` | Toggle comment |
| `F5` | Start debugging |
| `Ctrl + S` | Save file |

---

## 🎬 Complete Setup Flow

```bash
# 1. Open terminal in VS Code
Ctrl + `

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
python app.py

# 6. Open browser
http://localhost:5000
```

---

## 🎓 Understanding the Code Flow

1. **User opens browser** → `templates/index.html` loads
2. **User uploads files** → `app.py` `/upload` route saves files
3. **User clicks process** → `app.py` `/process` route:
   - Calls `face_utils.process_images()` → detects faces
   - Calls `clustering.cluster_faces()` → groups faces
   - Calls `organize_files()` → creates person folders
4. **Redirect to results** → `templates/results.html` shows stats
5. **User downloads** → `app.py` `/download` creates ZIP

---

## 📝 Making Changes

### To modify HTML/CSS:
- Edit files in `templates/` folder
- Refresh browser to see changes

### To modify Python logic:
- Edit `app.py`, `face_utils.py`, or `clustering.py`
- Stop app (Ctrl+C in terminal)
- Run again: `python app.py`

### To add new dependencies:
```bash
pip install package-name
pip freeze > requirements.txt  # Update requirements
```

---

## 🎯 Testing Your Changes

### Test face detection:
```python
# In Python terminal
from face_utils import process_images
import os

# Assuming you have test images in uploads/
files = os.listdir('uploads')
result = process_images('uploads', files)
print(f"Detected {len(result)} faces")
```

### Test clustering:
```python
from clustering import cluster_faces

# Using face_data from above
clustered = cluster_faces(result)
print(f"Found {len(set([f['cluster_id'] for f in clustered]))} groups")
```

---

## 🚨 If Something Goes Wrong

### App won't start:
```bash
# Check if port 5000 is in use
# Windows
netstat -ano | findstr :5000

# Mac/Linux
lsof -i :5000

# Kill the process or change port in app.py:
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Dependencies won't install:
```bash
# Try upgrading pip first
python -m pip install --upgrade pip

# Then retry
pip install -r requirements.txt
```

### Face recognition not working:
```bash
# Reinstall face_recognition
pip uninstall face_recognition dlib
pip install cmake
pip install dlib
pip install face_recognition
```

---

## ✅ Verification Checklist

Before testing:
- [ ] Virtual environment activated `(venv)` visible in terminal
- [ ] All dependencies installed (no errors in installation)
- [ ] Flask app running (shows "Running on http://...")
- [ ] Browser can access http://localhost:5000
- [ ] Have test images ready (photos with visible faces)

---

**Ready to go! 🎉 Start uploading photos and watch the magic happen!**
