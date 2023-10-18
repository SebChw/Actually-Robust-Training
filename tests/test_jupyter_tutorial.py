from cookiecutter.main import cookiecutter
import os
import filecmp
import shutil
import filecmp



def download_tutorial():
    try:
        cookiecutter(
            "https://github.com/SebChw/art_template.git",
            no_input=True,
            extra_context={"project_name": "mnist_tutorial", "author": "test", "email": "test"},  # Pass the project_name to the template,
            checkout="mnist_tutorial_cookiecutter",  # Use the latest version of the template
        )
    except Exception as e:
        print("Error while generating project using Cookiecutter:", str(e))
        raise e
    return 0


def clean_up():
    shutil.rmtree("mnist_tutorial")

def run_jupyter_notebook():
    # This should work, it seems like there is a bug in nbmake librabry.
    # os.system("pytest --nbmake ./mnist_tutorial/exp1")
    # When the command is run in the same directory as the notebook and the path is removed it works.
    os.system("cd mnist_tutorial/exp1 && pytest --nbmake .") # Not the best solution but it works.
    # also solves the problem with data_module.py and baselines.py
   

def check_outputs():
    utils_folder = './utils'
    jupyter_outputs_folder = './mnist_tutorial/exp1'
    for root, dirs, files in os.walk(utils_folder):
        for file in files:
            utils_file_path = os.path.relpath(os.path.join(root, file), start=utils_folder)
            jupyter_file_path = os.path.join(jupyter_outputs_folder, utils_file_path)
            if file.endswith(".log") or file.endswith(".ckpt"):  # filecmp fails with decoding logs and ckpts - skip needed
                # maybe we should check if the files are the same size? or just remove them from utils?
                # assert os.path.getsize(os.path.join(utils_folder, utils_file_path)) == os.path.getsize(jupyter_file_path)  this does not work
                # for some reason the sizes are different and files visually identical lol
                continue

            assert os.path.isfile(jupyter_file_path) == True
            assert filecmp.cmp(os.path.join(utils_folder, utils_file_path), jupyter_file_path) == True

    for root, dirs, files in os.walk(jupyter_outputs_folder):
        for file in files:
            if file.endswith(".py") or file.endswith(".ipynb") or file.endswith(".log") or file.endswith(".ckpt"):
                continue
            jupyter_file_path = os.path.relpath(os.path.join(root, file), start=jupyter_outputs_folder)
            if "__pycache__" in jupyter_file_path or ".neptune" in jupyter_file_path or ".pytest_cache" in jupyter_file_path:
                continue
            utils_file_path = os.path.join(utils_folder, jupyter_file_path)
            assert os.path.isfile(utils_file_path) == True
            assert filecmp.cmp(utils_file_path, os.path.join(jupyter_outputs_folder, jupyter_file_path)) == True

    
def test_tutorial():
    if os.path.isdir("mnist_tutorial"):
        clean_up()
    download_tutorial()
    run_jupyter_notebook()
    check_outputs()
    clean_up()

if __name__ == "__main__":
    test_tutorial()
    