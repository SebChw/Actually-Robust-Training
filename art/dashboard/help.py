from dash import dcc

HELP = dcc.Markdown(
    """
Within this dashboard you can track progress of your Project and visualize/compare results of each step.

On the left you see timeline with all steps that you've defined in your Project. 
* Next to every step there is a short summary of its results.
* Green indicates that step is passed and red indicates it failed.

On the right you can see table with all results of selected step.
* Select step with the dropdown selection. 
* You can sort and filter results by clicking on column headers. 
* You can also select single row to make it your `base` run. This will be important for the plot.

Below the table you can see plot with results of selected step.
* You can select which metrics you want to plot on x and y axis
* As you hover over the plot you can see logged details of each run
    * Your base run will be bigger and contain all logged metrics
    * Other runs will be smaller and contain only metrics that are different from the base run.
"""
)
