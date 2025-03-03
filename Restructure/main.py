#!/usr/bin/env python3



"""
Main entry point for the positioning system application.

This script provides a command-line interface to the positioning system,
allowing users to run different visualization and analysis modes.
"""


import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from config.load_config import load_config
from src.core.hardware import LEDSystem, PDSystem
from src.core.scenario import TestPoint
from src.core.solver_mulmul import Solver
from src.core.simulate_env import Environment
from src.visualization.plot import (
    plot_scenario, plot_analysis, plot_effective_range, plot_hardware_configuration
)
from src.visualization.interactive import (
    Interactive1To1Visualizer, InteractiveMultiVisualizer
)


def main():
    """Main entry point for the application."""

    # Load configuration
    config = load_config()
    mode = config["application"]["mode"]
    save_result = config["application"]["save_results"]
    
    # Initialize hardware systems
    led_system = LEDSystem(
        num=config["led_system"]["num"],
        hard_num=config["led_system"]["hard_ind"],
        config_num=config["led_system"]["config_num"],
        alpha=np.deg2rad(config["led_system"]["alpha"])
    )

    pd_system = PDSystem(
        num=config["pd_system"]["num"],
        hard_num=config["pd_system"]["hard_ind"],
        config_num=config["pd_system"]["config_num"],
        alpha=np.deg2rad(config["pd_system"]["alpha"])
    )

    # Create test points based on scenario
    scenario = config["scenario"]["type"]
    space_size = config["system"]["space_size"]
    rot_max = config["scenario"]["rot_max"]
    test_point = TestPoint(scenario, {"space_size": space_size})
    
    
    # Execute modes without the need of solver
    if mode == "scenario":
        # Draw scenario visualization
        fig = plot_scenario(test_point, scenario, space_size)
        
        if save_result:
            fig.savefig("data/scenario.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()
        return
    
    elif mode == "draw_config":
        # Draw hardware configuration
        fig = plot_hardware_configuration(led_system, pd_system)
        
        if save_result:
            fig.savefig("data/config.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()
        return



    # Initiate simulating environment
    env = Environment(led_system, pd_system, test_point, config)
    light_signals = env.light_output_filtered

    # Initialize solver
    solver = Solver(env, config)
    solver.solve_mulmul()
    
    
    
    
    # Execute requested mode
    if mode == "analysis":
        # Perform analysis and plot results
        solver.solve_mulmul()
        fig = plot_analysis(
            test_point, 
            solver.error, 
            solver.tolerance, 
            scenario, 
            rot_max, 
            space_size
        )
        
        print('finish analysis')
        if save_result:
            fig.savefig("data/analysis.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    elif mode == "interactive_1to1":
        # Launch interactive 1-to-1 visualization
        visualizer = Interactive1To1Visualizer(led_system, pd_system, solver)
        visualizer.show()
    
    elif mode == "interactive_mulmul":
        # Launch interactive multi-point visualization
        visualizer = InteractiveMultiVisualizer(
            led_system, 
            pd_system, 
            solver, 
            scenario, 
            space_size, 
            rot_max
        )
        visualizer.show()
    
    elif mode == "optimize":
        # Perform parameter optimization
        print("Optimizing system parameters...")
        # Implementation of optimization algorithm would go here
        # This is a placeholder for future implementation
        print("Optimization not yet implemented")


if __name__ == "__main__":
    main()