"""
Visualization utilities for the positioning system.

This module contains functions for generating various plots to visualize
the positioning system's performance and results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from typing import Dict, Tuple, List, Optional, Union
from src.core.scenario import TestPoint


def set_plot_style():
    """Set default plotting style for consistent appearance."""
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams['axes.unicode_minus'] = False


def plot_scenario(test_point: TestPoint, scenario: int, space_size: int = 10) -> plt.Figure:
    """
    Plot the test scenario showing sample points.
    
    Args:
        test_point: TestPoint instance with test positions and rotations
        scenario: Scenario number
        space_size: Size of space for scenario 3
    
    Returns:
        Figure object
    """
    set_plot_style()
    
    fig = plt.figure(figsize=(12, 8))
    
    # Plot translation samples
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_box_aspect(aspect=(1, 1, 1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid(True)
    ax.set_title('Translation Test Points Relative Positions')
    
    # Set axis limits based on scenario
    if scenario == 2:
        ax.set_xlim3d(-3, 3)
        ax.set_ylim3d(-3, 3)
        ax.set_zlim3d(-3, 3)
    elif scenario == 3:
        ax.set_xlim3d(-space_size/2, space_size/2)
        ax.set_ylim3d(-space_size/2, space_size/2)
        ax.set_zlim3d(0, space_size)
    else:
        ax.set_xlim3d(-1.5, 1.5)
        ax.set_ylim3d(-1.5, 1.5)
        ax.set_zlim3d(0, 3)

    # Plot sample points
    sc = ax.scatter(
        test_point.testp_pos[0, :],
        test_point.testp_pos[1, :],
        test_point.testp_pos[2, :],
        alpha=0.5,
        color='b',
        label='LED Coordinate System Position'
    )
    ax.scatter(0, 0, 0, color='r', marker='x', label='PD Coordinate System Position')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
    
    # Plot rotation samples in polar coordinates
    ax = fig.add_subplot(1, 3, 3, projection='polar')
    sc = ax.scatter(
        test_point.testp_rot[2, :],
        np.rad2deg(test_point.testp_rot[1, :]),
        color='b'
    )
    ax.set_title('Rotation Test Points (Pitch, Yaw)')
    ax.grid(True)
    
    # Plot rotation samples in 3D
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.set_title('Rotation Test Points Relative Orientations')
    ax.set_box_aspect(aspect=(1, 1, 1))
    ax.grid(False)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_axis_off()
    
    # Plot unit sphere for reference
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, np.pi, 20))
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="w", alpha=0.2, edgecolor="#808080")
    
    # Plot x-y plane
    a = np.linspace(-1, 1, 21)
    b = np.linspace(-1, 1, 21)
    A, B = np.meshgrid(a, b)
    c = np.zeros((21, 21))
    ax.plot_surface(A, B, c, color="grey", alpha=0.2)
    
    # Plot rotation vectors
    a, b, c1 = ori_ang2cart(test_point.testp_rot[1:, :])
    ax.quiver(0, 0, 0, 0, 0, -1, color='r', label='PD Coordinate System Z-axis')
    ax.quiver(0, 0, 0, a, b, c1, color='b', label='LED Coordinate System Z-axis')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1))
    
    return fig


def plot_analysis(test_point: TestPoint, error: np.ndarray, tolerance: float, 
                  scenario: int, rot_max: int = 180, space_size: int = 10) -> plt.Figure:
    """
    Plot analysis results showing error distribution.
    
    Args:
        test_point: TestPoint instance with test positions and rotations
        error: Error matrix [kpos x krot]
        tolerance: Error tolerance
        scenario: Scenario number
        rot_max: Maximum rotation angle for polar plots
        space_size: Size of space for scenario 3
    
    Returns:
        Figure object
    """
    set_plot_style()
    
    # Calculate counts of points within tolerance
    count_kpos = np.nansum(error < tolerance, axis=1)
    count_krot = np.nansum(error < tolerance, axis=0)
    
    fig = plt.figure(figsize=(12,5))
    # fig.tight_layout()
    colormap = plt.cm.get_cmap('YlOrRd')
    normalizep = colors.Normalize(vmin=0, vmax=test_point.krot)
    normalizer = colors.Normalize(vmin=0, vmax=test_point.kpos)
    # fig.subplots_adjust(wspace=0.3)
    
    # Plot translation samples with color indicating success rate
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_box_aspect(aspect=(1, 1, 1))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.grid(True)
    
    # Set axis limits based on scenario
    if scenario == 2:
        ax1.set_xlim3d(-3, 3)
        ax1.set_ylim3d(-3, 3)
        ax1.set_zlim3d(-3, 3)
    elif scenario == 3:
        ax1.set_xlim3d(-space_size/2, space_size/2)
        ax1.set_ylim3d(-space_size/2, space_size/2)
        ax1.set_zlim3d(0, space_size)
    else:
        ax1.set_xlim3d(-1.5, 1.5)
        ax1.set_ylim3d(-1.5, 1.5)
        ax1.set_zlim3d(0, 3)
    
    sc = ax1.scatter(
        test_point.testp_pos[0, :],
        test_point.testp_pos[1, :],
        test_point.testp_pos[2, :],
        c=count_kpos,
        cmap=colormap,
        norm=normalizep,
        alpha=0.5
    )
    ax1.scatter(0, 0, 0, color='k', marker='x')
    
    colorbar = fig.colorbar(sc, shrink=0.3, pad=0.15)
    colorbar.ax.set_ylabel('# Sample Points within Tolerance')
    ax1.set_title('Translation Sample Point')


    ax2 = fig.add_subplot(1,2,2,projection='polar')
    sc = ax2.scatter(test_point.testp_rot[2,:],np.rad2deg(test_point.testp_rot[1,:])  ,c = count_krot,cmap=colormap,norm = normalizer)
    
    colorbar = fig.colorbar(sc,shrink=0.3,pad=0.15)
    colorbar.ax.set_ylabel('# Sample Points within Tolerance')
    ax2.set_title('Rotation Sample Points')
    ax2.text(np.deg2rad(330), (15),'pitch(degree)',rotation = 15)
    ax2.text(np.deg2rad(60), 190 ,'yaw(degree)')
    
    # Add summary stats
    count_total = np.nansum(error < tolerance)
    fig.suptitle(f'Total samples within tolerance: {count_total}')
    
    return fig


def plot_effective_range(test_point: TestPoint, error: np.ndarray, tolerance: float, effective: float,
                        scenario: int, rot_max: int = 180, space_size: int = 10) -> plt.Figure:
    """
    Plot effective range showing where positioning is reliable.
    
    Args:
        test_point: TestPoint instance with test positions and rotations
        error: Error matrix [kpos x krot]
        tolerance: Error tolerance
        effective: Effective percentage threshold
        scenario: Scenario number
        rot_max: Maximum rotation angle for polar plots
        space_size: Size of space for scenario 3
    
    Returns:
        Figure object
    """
    set_plot_style()
    
    # Calculate counts and effective regions
    count_kpos = np.nansum(error < tolerance, axis=1)
    count_krot = np.nansum(error < tolerance, axis=0)
    effective_pos = count_kpos / test_point.krot >= effective / 100
    effective_rot = count_krot / test_point.kpos >= effective / 100
    
    fig = plt.figure(figsize=(15, 8))
    colormap = plt.cm.get_cmap('YlOrRd')
    normalizep = colors.Normalize(vmin=0, vmax=test_point.krot)
    normalizer = colors.Normalize(vmin=0, vmax=test_point.kpos)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Plot all translation samples with color
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.set_box_aspect(aspect=(1, 1, 1))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.grid(True)
    
    # Set axis limits based on scenario
    if scenario == 2:
        ax1.set_xlim3d(-3, 3)
        ax1.set_ylim3d(-3, 3)
        ax1.set_zlim3d(-3, 3)
    elif scenario == 3:
        ax1.set_xlim3d(-space_size/2, space_size/2)
        ax1.set_ylim3d(-space_size/2, space_size/2)
        ax1.set_zlim3d(0, space_size)
    else:
        ax1.set_xlim3d(-1.5, 1.5)
        ax1.set_ylim3d(-1.5, 1.5)
        ax1.set_zlim3d(0, 3)
    
    sc1 = ax1.scatter(
        test_point.testp_pos[0, :],
        test_point.testp_pos[1, :],
        test_point.testp_pos[2, :],
        c=count_kpos,
        cmap=colormap,
        norm=normalizep,
        alpha=0.5
    )
    ax1.scatter(0, 0, 0, color='k', marker='x')
    
    colorbar = fig.colorbar(sc1, shrink=0.3, pad=0.15)
    ax1.set_title('Translation Samples')
    colorbar.ax.set_ylabel('Count of Samples within Tolerance')
    
    # Plot all rotation samples with color
    ax2 = fig.add_subplot(2, 3, 4, projection='polar')
    sc2 = ax2.scatter(
        test_point.testp_rot[2, :],
        np.rad2deg(test_point.testp_rot[1, :]),
        c=count_krot,
        cmap=colormap,
        norm=normalizer
    )
    colorbar = fig.colorbar(sc2, shrink=0.3, pad=0.15)
    colorbar.ax.set_ylabel('Count of Samples within Tolerance')
    ax2.set_title('Rotation Samples')
    ax2.text(1, 1, 'pitch (degrees)', rotation=15)
    ax2.text(np.deg2rad(60), 80, 'yaw (degrees)')
    ax2.set_ylim([0, rot_max])
    
    # Plot effective translation samples
    ax3 = fig.add_subplot(2, 3, 2, projection='3d')
    ax3.set_box_aspect(aspect=(1, 1, 1))
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.grid(True)
    
    # Set axis limits based on scenario
    if scenario == 2:
        ax3.set_xlim3d(-3, 3)
        ax3.set_ylim3d(-3, 3)
        ax3.set_zlim3d(-3, 3)
    else:
        ax3.set_xlim3d(-1.5, 1.5)
        ax3.set_ylim3d(-1.5, 1.5)
        ax3.set_zlim3d(0, 3)
    
    sc3 = ax3.scatter(
        test_point.testp_pos[0, effective_pos],
        test_point.testp_pos[1, effective_pos],
        test_point.testp_pos[2, effective_pos],
        color='b',
        alpha=0.5
    )
    ax3.scatter(0, 0, 0, color='k', marker='x')
    ax3.set_title(f'Effective Translation Range\n({effective}% success)')
    
    # Plot effective rotation samples
    ax4 = fig.add_subplot(2, 3, 5, projection='polar')
    sc4 = ax4.scatter(
        test_point.testp_rot[2, effective_rot],
        np.rad2deg(test_point.testp_rot[1, effective_rot]),
        color='b'
    )
    ax4.set_title(f'Effective Rotation Range\n({effective}% success)')
    ax4.text(0, 0, 'pitch (degrees)', rotation=15)
    ax4.text(np.deg2rad(60), 80, 'yaw (degrees)')
    ax4.set_ylim([0, rot_max])
    
    # Add summary stats
    count_total = np.nansum(error < tolerance)
    fig.suptitle(f'Total samples within tolerance: {count_total}')
    
    return fig


def plot_hardware_configuration(led_system, pd_system):
    """
    Plot hardware configuration showing orientations of LEDs and PDs.
    
    Args:
        led_system: LEDSystem instance
        pd_system: PhotodiodeSystem instance
        
    Returns:
        Figure object
    """
    set_plot_style()
    
    fig = plt.figure(figsize=(12, 6))
    
    # Plot LED orientation
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.zaxis.set_ticklabels([])
    ax1.set_axis_off()
    ax1.set_box_aspect(aspect=(1, 1, 1))
    ax1.set_title('LED System')
    ax1.set_xlim3d(-1, 1)
    ax1.set_ylim3d(-1, 1)
    ax1.set_zlim3d(-1, 1)
    
    # Draw coordinate axes
    ax1.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color='grey')
    ax1.text(1.1, 0, 0, 'x', color='grey')
    ax1.text(0, 1.1, 0, 'y', color='grey')
    ax1.text(0, 0, 1.1, 'z', color='grey')
    
    # Draw x-y plane
    a = np.linspace(0, 2*np.pi, 20)
    b = np.linspace(0, 1, 10)
    r = np.outer(b, np.cos(a))
    o = np.outer(b, np.sin(a))
    zeror = np.zeros(r.shape)
    ax1.plot_surface(r, o, zeror, color="grey", alpha=0.25)
    
    # Draw unit sphere
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 20), np.linspace(0, np.pi, 21))
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax1.plot_wireframe(x, y, z, color="w", alpha=0.15, edgecolor="#808080")
    
    # Draw LED orientations
    zeros = np.zeros((led_system.num,))
    ax1.quiver(zeros, zeros, zeros, 
               led_system.ori_car[0, :], 
               led_system.ori_car[1, :], 
               led_system.ori_car[2, :], 
               color='b', label='LED Orientation')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2))
    
    # Plot PD orientation
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])
    ax2.zaxis.set_ticklabels([])
    ax2.set_axis_off()
    ax2.set_box_aspect(aspect=(1, 1, 1))
    ax2.set_title('Photodiode System')
    ax2.set_xlim3d(-1, 1)
    ax2.set_ylim3d(-1, 1)
    ax2.set_zlim3d(-1, 1)
    
    # Draw coordinate axes
    ax2.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color='grey')
    ax2.text(1.1, 0, 0, 'x', color='grey')
    ax2.text(0, 1.1, 0, 'y', color='grey')
    ax2.text(0, 0, 1.1, 'z', color='grey')
    
    # Draw x-y plane
    ax2.plot_surface(r, o, zeror, color="grey", alpha=0.25)
    
    # Draw unit sphere
    ax2.plot_wireframe(x, y, z, color="w", alpha=0.15, edgecolor="#808080")
    
    # Draw PD orientations
    zeros = np.zeros((pd_system.num,))
    ax2.quiver(zeros, zeros, zeros, 
               pd_system.ori_car[0, :], 
               pd_system.ori_car[1, :], 
               pd_system.ori_car[2, :], 
               color='firebrick', label='PD Orientation')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2))
    
    return fig
    ax.set_title('Translation Samples')
    colorbar.ax.set_ylabel('Count of Samples within Tolerance')
    
    # Plot rotation samples with color indicating success rate
    ax = fig.add_subplot(1, 3, 2, projection='polar')
    sc = ax.scatter(
        test_point.testp_rot[2, :],
        np.rad2deg(test_point.testp_rot[1, :]),
        c=count_krot,
        cmap=colormap,
        norm=normalizer
    )
    colorbar = fig.colorbar(sc, shrink=0.3, pad=0.15)


def ori_ang2cart(ori_ang):#ori_ang = 2xsensor_num np.array, 第一列傾角 第二列方位
    return np.stack((\
    np.multiply(np.sin(ori_ang[0,:]), np.cos(ori_ang[1,:])),\
    np.multiply(np.sin(ori_ang[0,:]), np.sin(ori_ang[1,:])),\
    np.cos(ori_ang[0,:])    \
        ),0)