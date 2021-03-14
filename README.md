# simple machine learning web application

To create conda based python virtual environment first open new terminal in project's directory. Then type `$ conda env create -f environment.yml` and when done - activate the new environment.
  - Tensorflow venv with GPU:
  
`$ conda create --name tf2`

`$ conda activate tf2`

`$ conda install tensorflow-gpu`
  - If you prefer pipenv:

`export PIPENV_VENV_IN_PROJECT="enabled"`

`pipenv --three`

`pipenv shell`

`pip install streamlit`

`streamlit hello`
