init:
	pip3 install -r requirements.txt

test:
	python3 -m unittest scripts.tests.test_extract_workflow_connections
