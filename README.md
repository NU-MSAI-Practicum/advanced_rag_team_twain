# advanced_rag_team_twain

## Local Development Setup
1. Open a terminal and navigate to your project folder.

`$ cd advanced_rag_team_twain`

2. In your terminal, type:

`$ python -m venv .venv`

3. A folder named ".venv" will appear in your project. This directory is where your virtual environment and its dependencies are installed.


4. In your terminal, activate your environment with one of the following commands, depending on your operating system.

`$ source .venv/bin/activate`

5. Download necessary packages

`$ pip install -r requirements. txt`

6. Run your Streamlit app.

`$ python -m streamlit run app.py`

7. To stop the Streamlit server, press `Ctrl+C` in the terminal.


8. When you're done using this environment, return to your normal shell by typing:

`$ deactivate`

## Local Docker Setup

1. Generate requirements, (pipreqs only generate requirements for the current project)

`$ pip install pipreqs`

`$ pipreqs . --ignore ".venv" `

2. Build image

`$ docker build -t your-image-name:tag .`

3. Run image

`$ docker run -d --name your-container-name -p 8501:8501 -v ~/.aws:/root/.aws your-image-name:latest`

4. Visit localhost:8501 in the browser

5. Cleanup work: stop container

`$ docker stop test-container`

6. Cleanup work: delete container

`$ docker rm test-container`

7. Cleanup work: delete image

`$ docker rmi <image_name>:<tag> `