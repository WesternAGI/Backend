

First, you need to clone this repository. 

Second, you need to create a virtual environment using the requirements.txt file and activate the environment. 

To test the server locally:
1: From the main directory (Backend), run the server locally:python -m server.main
2: Run the test script locally: TEST_LOCAL=1 pytest -s test1.py 

To test the deployed server (which is running on Railway.app)
1: Run the test script: pytest -s test1.py 
