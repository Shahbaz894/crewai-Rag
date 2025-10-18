import sys
import os
from crew import Crew,Agent,Task
import yaml
# Ensure Python can find the 'logs' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from logs.logger import log_info, log_error, log_debug, log_exception



with open("../config/agents.yaml") as f:
    agents_config = yaml.safe_load(f)
    log_info('load the agent yaml file')

with open("../config/tasks.yaml") as f:
    tasks_config = yaml.safe_load(f)
    log_info('load the agent task file')
    
    
# Define agents
medical_researcher = Agent(**agents_config["medical_researcher"])
log_info('medical researecher create successfull')
medical_analyst = Agent(**agents_config["medical_analyst"])

# Define tasks
research_task = Task(**tasks_config["research_task"])
reporting_task = Task(**tasks_config["reporting_task"])

crew =Crew(agents=[medical_researcher,medical_analyst],tasks=[research_task,reporting_task]
           
           
           )