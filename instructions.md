## Python environment installation instructions
- Make sure you have conda installed in your system. [Instructions link here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).
- Then, get the `conda_env.yml` file, and from the same directory, run `conda env create -f conda_env.yml`. 
- Activate the environment - `conda activate ddrl`
- Go into environment directory - `cd particle-envs`
- Install the environment - `pip install -e .`

## Running the code
- Make sure you have the environment activated, and you are in the `policy` directory.
- Complete the code in `dagger_template.py`.
- Command to run code: `python dagger_template.py`.
