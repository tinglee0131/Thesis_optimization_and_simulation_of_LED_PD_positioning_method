This is for refactoring and restructuring my thesis code.


# Plan
- Docker containerizatoin
- Configuration seperated from executing file, using maybe yaml or json
- Seperated the code in structural way instead of single file
    - visualization: for the graphs and GUI
    - engine? core? for the base of the system with calculating
    - func: for some functions defined(utils for convention naming?)

- coding style problem
    - no OOP
    - too many globals(should slow down performance)
    - Python function new style:
        - Typing: New extension for indicating types of input and output, convenient to check to prevent error.
            - -> indicate return type
        - No need of \ for seperating a long line into multiple lines
        
- Docker Notes
    - useful tips: 
        - bind mound the container's data to the local machine
            - why: container will not keep the data. Data only exists in container.
            - how: use the -v option in docker run
        - figure showing problem
            - linux slim does not have the system file for matplotlib. need to apt-get
            - forward display to local machine
                - xhost(mac use XQuantz): CLI tool controling access of X11 display server
                    - '''xhost +local:docker'''
                    - giving docker access to the local X server
                - X11 display server: Unix-like OS manages GUI
- OOP notes
    - passing instance can keep the consistency of value. just take value from instance everytime

