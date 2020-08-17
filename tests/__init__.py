import os
import sys


project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
api_path= os.path.join(project_path, 'api')
sys.path.append(api_path)
