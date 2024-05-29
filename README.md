# advanced_rag_team_twain
This is the branch for deployment code. Any push will automatically trigger Github Action CICD to containerize the repo, push the image to ECR, and deploy it on EC2.

## Directory

- .github/workflows/: the deployment file for github actions: containerize, push the image to ECR, deploy on EC2

- config/: app's central configuration

- scripts/: data ingestion scripts: insert knowledge base to Elasticsearch in bulk

- utils/: utilities

- Dockerfile: the docker file during containerization

- app.py: starting script for streamlit demo

- chatbot.py: some functions to generate bot responses

## Environment Requirement

Python >= 3.9


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

## Deployment steps
1. Apply for an IAM user with `Opensearchfullaccess` and `ECRfullaccess`


2. Create an EC2 instance and generate a key pair for SSH 
- Please meet llama3 8B's basic hardware requirement (We are using AWS g4dn.2xlarge)
- Please choose a pytorch version with cuda and pytorch preinstalled

3. Create an ECR registry repo to store images
   
During configuration, please make the image is immutable!!

4. Try SSH to the server (Follow this tutorial: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-linux-inst-ssh.html)

(1) `$ chmod 400 key-pair-name.pem`

(2) `$ ssh -i /path/key-pair-name.pem instance-user-name@instance-public-dns-name`


5. SSH to your machine to make initial configurations.

(1) Install docker (Follow this tutorial: https://medium.com/@srijaanaparthy/step-by-step-guide-to-install-docker-on-amazon-linux-machine-in-aws-a690bf44b5fe)

(2) Add your user to the Docker group to run Docker commands without 'sudo': 

`$ sudo usermod -a -G docker ec2-user`
   
After adding the user to the Docker group, simply disconnect and reconnect to SSH to refresh the group memberships.
   
(3) Configure AWS credentials on the server: `$ aws configure`, Input access key, secret key of the IAM user

(4) Log on to ECR on the server: please replace `123456789012.dkr.ecr.us-east-2.amazonaws.com/myapp` with your ECR repository

`$ aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-2.amazonaws.com/myapp`

(5) Install Ollama(GPU version)
- Install GPU toolkit, follow this tutorial: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation
- Download ollama docker image: `$ docker pull ollama/ollama` (Reference: https://hub.docker.com/r/ollama/ollama)
- Run ollama gpu version: `$ docker run -d --rm --gpus=all -v /home/ollama:/root/.ollama:z -p 11434:11434 --name ollama ollama/ollama` (Reference: https://medium.com/@blackhorseya/running-llama-3-model-with-nvidia-gpu-using-ollama-docker-on-rhel-9-0504aeb1c924)
After this, it is not using GPU in nvidia-smi)
- Download model: 
`$ docker exec -it ollama ollama pull llama3`
- List model and run the desired one
`$ docker exec -it ollama ollama list`
`$ docker exec -it ollama ollama run llama3`
- Type /bye if it is an interactive session
- Test the setup
`$ curl http://localhost:11434/api/generate -d '{
      "model": "llama3",
      "prompt": "請列出五樣台灣美食",
      "stream": true,
      "options": {
      "seed": 123,
      "top_k": 20,
      "top_p": 0.9,
      "temperature": 0
      }
      }'`

（6）Install Ollama (CPU version: NOT RECOMMENDED for production! A single generation can take 15 minutes.)
- Download Ollama: `$ curl -fsSL https://ollama.com/install.sh | sh`
- Pull llama3 model: `$ ollama pull llama3`
- Test if ollama is running properly
`$ curl http://localhost:11434/api/chat -d '{
          "model": "llama3",
          "messages": [{ "role": "user", "content": "Are you a robot?" }],
          "stream": false
      }'`
- Wait for some time, you should get: 
      `$ {"model":"openhermes2.5-mistral", "created_at":"2024-01-30T11:52:56.244775Z", "message":{"role":"assistant", "content":"No, I'm not a robot. I am an AI-powered chatbot designed to provide helpful information and engage in conversation with users like yourself. 😊}, "done":true, "total_duration":22872800167, "load_duration":7669185084, "prompt_eval_count":21, "prompt_eval_duration":284746000, "eval_count":195,"eval_duration":14899744000}`
- Allow all visits to the ollama service: edit this file: `$ sudo vim /etc/systemd/system/ollama.service`
      
      [Unit]
      Description=Ollama Service
      After=network-online.target

      [Service]
      ExecStart=/usr/local/bin/ollama serve
      User=ollama
      Group=ollama
      Restart=always
      RestartSec=3
      Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
      Environment="OLLAMA_HOST=0.0.0.0:11434"
      Environment="OLLAMA_ORIGINS=http://0.0.0.0:11434"

      [Install]
      WantedBy=default.target
      Basically its adding two lines
      Environment="OLLAMA_HOST=0.0.0.0:11434"
      Environment="OLLAMA_ORIGINS=http://0.0.0.0:11434"

- Reload the daemon and the service:

`$ sudo systemctl daemon-reload `

`$ sudo systemctl restart ollama.service`

6. Go to Github-project page - Settings - Secrets, add below key value pairs so that Github Actions would be able to fetch these configurations
   - AWS_ACCESS_KEY_ID: in the IAM credentials
   - AWS_SECRET_ACCESS_KEY: in the IAM credentials
   - HOST: EC2 public ipv4 dns
   - PORT: SSH port, 22 by default
   - SSH KEY: The generated ssh key pair: Please copy the content of xxx.pem
   - USERNAME: ec2-user by default for Amazon Linux OS, other systems please refer to https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-to-linux-instance.html#connection-prereqs-private-key