# bone-fracture-detection
Detection and recognition of bone fractures using a machine learning model with marimo notebooks

## Setup
### 1. Install git
check if installed:
```
git -v
```

### 2. Install Python and pip or uv
if you are using pip make shure to have python 3.12 or lower installed since 3.13 isn't working yet with some packages. If you are using uv it doesn't matter.

[python installation](https://www.python.org/downloads/) \
[uv installation](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)

check if both are installed:
```
python --version

pip --version  // or with uv:
uv --version
```

### (*) Install Python certificates on MacOS
If you are on Mac you need to install certain Python certifacates.
Do this by navigating the given path in Finder and then double click on Install certificates.command:
```
macOS Macintosh HD > Applications > Python3.12 > Install Certificates.command
```
It should open a terminal window. If all is done without an error you can close this terminal.

### 3. Clone the Repository
```
git clone https://github.com/DaafWedaPivat/bone-fracture-detection.git
cd bone-fracture-detection
```

### 4. Add dependencies
Download the [FracAtlas Dataset](https://figshare.com/articles/dataset/The_dataset/22363012) and extract it in private/dependencies

### 5. Create a virtual environment
#### with pip
make sure to not use python 3.13 and cd in private/dependencies
```
python -m venv venv  // or if your standard installation is 3.13:
python3.12 -m venv venv
```

#### with uv
```
uv venv --python 3.12
```

### 6. Activate environment

| Platform      | Shell       | Command to Activate Virtual Environment                             |
|---------------|-------------|---------------------------------------------------------------------|
| (Mac/Linux)   | bash/zsh    | `$ source private/dependencies/venv/bin/activate`                   |
|               | fish        | `$ source private/dependencies/venv/bin/activate.fish`              |
|               | csh/tcsh    | `$ source private/dependencies/venv/bin/activate.csh`               |
|               | pwsh        | `$ private/dependencies/venv/bin/Activate.ps1`                      |
| Windows       | cmd.exe     | `C:\> private\dependencies\venv\Scripts\activate.bat`               |
|               | PowerShell  | `PS C:\> private\dependencies\venv\Scripts\Activate.ps1`            |

### 7. Install neccessary python packages
Install all required python packages by installing the requirements.txt:

with pip:
'pip install -r requirements.txt'

with uv:
'uv pip install -r requirements.txt'

### 8. Download trained models
'mkdir private/generated/yolov8l400/weights && curl -L "https://www.dropbox.com/scl/fi/vf7g0q02f4wuzf7puqrje/best.pt?rlkey=lnm2apm2lusnf5bvtwnn5ns3c&st=7tu4upve&dl=1" -o private/generated/yolov8l400/weights/best.pt'

### 9. Run the dataset and after that the ui script
'python3.12 private/make/yolo_dataset.py && cd private/src/ui/'
After this:
'streamlit run ./'web_ui_streamlit.py
