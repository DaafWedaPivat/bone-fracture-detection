# bone-fracture-detection
Detection and recognition of bone fractures using a machine learning model with marimo notebooks

## Setup
### 1. Install git
check if installed:
```
git -v
```

### 2. Install Python and pip ur uv
if you are using pip make shure to have python 3.12 or lower installed since 3.13 isn't working yet with some packages. If you are using uv it doesn't matter.

[python installation](https://www.python.org/downloads/) \
[uv installation](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)

check if both are installed:
```
python --version

pip --version  // or with uv:
uv --version
```

### 3. Clone the Repository
```
git clone https://github.com/DaafWedaPivat/bone-fracture-detection.git
cd bone-fracture-detection
```

### 4. Create a virtual environment
#### with pip
make sure to not use python 3.13
```
python -m venv .venv  // or if your standard installation is 3.13:
python3.12 -m venv .venv
```

#### with uv
```
uv venv --python 3.12
```

### 5. Activate environment

| Platform | Shell       | Command to Activate Virtual Environment       |
|----------|-------------|-----------------------------------------------|
| POSIX    | bash/zsh    | `$ source .venv/bin/activate`                |
|          | fish        | `$ source .venv/bin/activate.fish`           |
|          | csh/tcsh    | `$ source .venv/bin/activate.csh`            |
|          | pwsh        | `$ .venv/bin/Activate.ps1`                   |
| Windows  | cmd.exe     | `C:\> .venv\Scripts\activate.bat`            |
|          | PowerShell  | `PS C:\> .venv\Scripts\Activate.ps1`         |

### 6. Start marimo
```
// only the first time
pip install marimo  // or with uv:
uv pip install marimo

// to start marimo
marimo edit
```
If you run it for the first time, marimo will ask in a popup if it should install all needed packages. Choose either pip or uv and click install.
