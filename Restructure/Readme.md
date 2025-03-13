# Positioning System - Restructured



## Overview

This project implements an visible light positioning system using LEDs and photodiode(PD) for versatile relative positioning. The system calculates the relative position and orientation between LED and PD coordinate by analyzing the light intensity measurements between multiple LEDs and PDs.

### Why Restructuring
- poor encapsulation: global variables for passing data, was not packaged and confusing -> OOP object
- poor structure: everything in the same code, which is hard for reading and developing -> structured
- poor mobility: the code was hard to deploy on other machine -> containerized
- poor input method: was changing the input in the code -> use config
- poor readability: code is sequential without proper function or object wrapping -> resturcture
- mixing Eng and Ch: the comment and everything is mixing in 2 languages, maybe sticking to Eng is better


## Project Structure
- Restructure
    - config/
        - config.yaml
        - load_config.py
    - data/ # for storing output
    - src/ 
        - core/
            - hardware.py # LED and PD component
            - scenario.py # generate testing scenario
            - simulate_env.py # simulate the PD signals
            - solver_mulmul.py # implement algorithm to solve the relative position
        - visualization/
            - plot.py # static images
            - interactive.py # interactive figure with sliding bars
    - Dockerfile # docker container
    - requirements.txt # python packages needed
    - main.py # main system entry
### Workflow
- setting needed info
    - hardware.py: LEDSystem, PDSystem
    - scenario.py: TestPoint
- simulate_env: take instance of LEDSystem, PDSystem, TestPoint to simulate the signals of PD read data
- solver_mulmul: calculate the position from PD data measured(simulated) from Environment instance, and the config from LEDSystem and PDSystem
- => For in teractive figure, each time needs to reset the info, and simulate the PD data in Environment instance, then calculate the position from Solver instance


## Deploy

### Using Docker

1. Build the Docker image:
   cd to the Restructure/ dir 
   - interactive
    ```bash
    docker build -t --it restructure-interact .
    ```
    should set the Entrypoint in Dockerfile to the /bin/bash/
   - run once
    ```bash
    docker build -t restructure-static .
    ```
    should set the Entrypoint in Dockerfile to the main.py

2. Run the container:
    - interactive developing
        
        ```bash
        docker run -it   -v "$(pwd):/app" -e DISPLAY=host.docker.internal:0 --rm  restructure-interact
        ```

        - volume bind mount to local machine: enable the local IDE while saving changes in container
        - display forward to local machine: for showing interactive figures
            - change container display to the host.docker.internal(a special domain name), and the number of display is 0(default map to 6000 port). The xhost is default to be listening at 6000 port.
            - (Possible:) might need to set the host machine's authentication of xhost to be access by docker.
                ```xhost +local:docker```
                
  




## Configuration
How: edit `config/default_config.yaml`

