# How to use the environment
1. Download Docker desktop for your device.
2. Open the command line terminal and create a new container(everything is done via CML for the container).
3. Running in the Docker container according to this official link. https://github.com/google-research/football/blob/master/gfootball/doc/docker.md 
4. If step3 doesn’t work, you can replace the Dockerfile in the container with the one provided in WhatsApp.
5. When running, there might be more packages to be installed. No worries, it will be very simple!  Just use pip install. As far as I know, we should at least pip install six, and pip install stable-baselines3==1.7.0(be careful with the right version). if not enough, Install the package as instructed by the command line interface.
6. Model outputs can show if you run the code in this project. The reward = plus 1 reward with 1 successful  goal + 1 reward approaching the opponent’s goal area(0.7 reward means 7/10 approaching in terms of Euclidean distance ) .  I evaluate this model per 10000 env steps, by 5 games in each evaluation. You can try your own settings. 

