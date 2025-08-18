This is the subfolder created for the project: Building an Artificial Intelligence Model of Post Traumatic Stress Disorder.
## Usage
1. Navigate to the blue_ai folder in your terminal
2. If you wish to run a whole trial with all the implemented functionalities, go to the execute.py script in the experiments folder and type the following command in your terminal:
    ```bash
        python -m blue_ai.scripts.ptsd.experiments.execute <directory>
    ```
    For 'directory' insert the name of a folder, which will be created to save the results and plots.
3. It is also possible to run the single scripts independently. Simply type:
    ```bash
        python -m blue_ai.scripts.ptsd.<script_name>
    ```
    followed by the necessary system arguments.