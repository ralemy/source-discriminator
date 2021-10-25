## Deploying with Docker and Docker compose
* Create a project directory.
* Inside it, create a persist directory and a container directory.
* Put docker-compose.yml file in the project directory
* Put Dockerfile in the container directory
* Go to persist directory and clone the repository there.
* make sure all folders (logs, plots, feature_store, etc.) are created and accessible as defined in config.ini file
  * For example, if using the provided config.ini file, create a logs/gradient_tape, plots/, feature_store/, data (with training data), etc in the persist directory.

Then bring up the docker compose with:
'''
docker-compose up
'''

once it is up, you can get the id to the container using:
'''
docker ps
'''

and once you know the container, you can interact with it:
'''
docker exec -it <containerid> /bin/bash
'''

once inside the container, you can run the app

'''
python main.py --action train --config config.ini
'''