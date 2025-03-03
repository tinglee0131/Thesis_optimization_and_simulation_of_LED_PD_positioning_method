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
        