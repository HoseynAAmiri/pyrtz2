# pyrtz2

Analysis of AFM force curves in Python.

Developed at Georgia Institute of Technology

# Installation
pyrtz2 is on PyPI. Install using pip (Python version >= 3.10.0 is required)

```
pip install pyrtz2
```

Please see the example folder. To run the HTML dash app interface simply use:

```
from pyrtz2 import app
app.run()
```
You should see this interface:

![pyrtz2.app](./example/con050.PNG)

You can select the contact point interactively. It will perform fits for approach and dwell parts of the curves using Hertzian and biexponential equations. After downloading the `csv` of fits, you can download those curves in one `pdf` file.

These options are under development:
- Show Fits
- Download Image Data
- Download Experiment
