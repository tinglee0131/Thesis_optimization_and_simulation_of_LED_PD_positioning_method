"""
Interactive1To1Visualizer
InteractiveNultiVisualizer

Provide interactive figure for users to visualize the result from simulation using sliders to adjust the configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.colors as colors
from typing import Dict, Optional

from src.core.hardware import LEDSystem, PDSystem
from src.core.scenario import TestPoint
from src.core.solver_mulmul import Solver


class Interactive1To1Visualizer:
    """
    Interactive visualizer for 1-to-1 positioning scenarios.
    
    This visualizer allows users to adjust the relative position and orientation
    of the LED and PD systems, as well as various parameters, and see the
    positioning results in real-time.
    """
    
    def __init__(self, led_system: LEDSystem, pd_system: PDSystem, solver: Solver):
        """
        Initialize the interactive visualizer.
        
        Args:
            led_system: LED system instance
            pd_system: PD system instance
            solver: Positioning solver instance
        """
        self.led_system = led_system
        self.pd_system = pd_system
        self.solver = solver
        self.env = solver.env
        self.testp = self.env.testp
        
        # Default test point for 1-to-1 visualization
        self.testp.testp_pos = np.array([[.3, 0.3, 1.5]]).T
        self.testp.testp_rot = np.array([[np.pi, 0, 0]]).T
        
        # Initialize figure and axes
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(121, projection='3d')
        self.setup_plot()
        
        # Initialize sliders and widgets
        self.sliders = []
        self.setup_sliders()
        
        # Solve initial configuration
        self.solve_and_update()
    
    def setup_plot(self):
        """Setup the 3D plot for visualizatsion."""
        self.ax.set_box_aspect(aspect=(1, 1, 1))
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.grid(True)
        self.ax.set_xlim3d(-1.5, 1.5)
        self.ax.set_ylim3d(-1.5, 1.5)
        self.ax.set_zlim3d(0, 3)
        
        # Adjust layout to make room for sliders
        self.fig.subplots_adjust(left=0.25, bottom=0.25)
        
        # Draw reference sphere at LED position
        u, v = np.meshgrid(np.linspace(0, 2*np.pi, 20), np.linspace(0, np.pi, 20))
        x = 0.5 * np.cos(u) * np.sin(v)
        y = 0.5 * np.sin(u) * np.sin(v)
        z = 0.5 * np.cos(v)
        self.sphere = self.ax.plot_wireframe(
            x + self.testp.testp_pos[0, 0],
            y + self.testp.testp_pos[1, 0],
            z + self.testp.testp_pos[2, 0],
            color="w", alpha=0.2, edgecolor="#808080"
        )
        self.ref_sphere = self.ax.plot_wireframe(
            x, y, z, color="w", alpha=0.2, edgecolor="#808080"
        )
        
        # Draw coordinate systems
        arrow = 0.5 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
        self.ax.quiver(
            np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]),
            arrow[0, :], arrow[1, :], arrow[2, :],
            arrow_length_ratio=0.2, color='firebrick', label='PD Coordinate System'
        )
        
        arrow_rot = self.env.rotate_mat(self.testp.testp_rot.flatten()) @ arrow
        self.axis_item = self.ax.quiver(
            self.testp.testp_pos[0, 0], self.testp.testp_pos[1, 0], self.testp.testp_pos[2, 0],
            arrow_rot[0, :], arrow_rot[1, :], arrow_rot[2, :],
            arrow_length_ratio=0.1, color='b', label='LED Coordinate System'
        )
        
        # Add coordinate labels
        self.ax.text(0, 0, 0.5, 'z', color='r')
        self.ax.text(0, 0.5, 0, 'y', color='r')
        self.ax.text(0.5, 0, 0, 'x', color='r')
        
        self.led_text = [None, None, None]
        self.led_text[0] = self.ax.text(
            self.testp.testp_pos[0, 0] + 1.1 * arrow_rot[0, 0],
            self.testp.testp_pos[1, 0] + 1.1 * arrow_rot[1, 0],
            self.testp.testp_pos[2, 0] + 1.1 * arrow_rot[2, 0],
            'x', color='b'
        )
        self.led_text[1] = self.ax.text(
            self.testp.testp_pos[0, 0] + 1.1 * arrow_rot[0, 1],
            self.testp.testp_pos[1, 0] + 1.1 * arrow_rot[1, 1],
            self.testp.testp_pos[2, 0] + 1.1 * arrow_rot[2, 1],
            'y', color='b'
        )
        self.led_text[2] = self.ax.text(
            self.testp.testp_pos[0, 0] + 1.1 * arrow_rot[0, 2],
            self.testp.testp_pos[1, 0] + 1.1 * arrow_rot[1, 2],
            self.testp.testp_pos[2, 0] + 1.1 * arrow_rot[2, 2],
            'z', color='b'
        )
        
        # Add placeholders for solution and error visualization
        self.ax.quiver(0, 0, 0, 0, 0, 0, color='k', label='Computed Relative Position')
        self.ax.quiver(0, 0, 0, 0, 0, 0, color='magenta', label='Error')
        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
        
        # Add text for results
        # self.text_item = self.ax.text(-2.5, -2.5, -2, 'Solving...')
        
        # Placeholders for solution visualization
        self.ans = None
        self.error_vec = None
        self.text_item = None

        pdu = self.solver.pdu[0,0]
        ledu = self.solver.ledu[0,0]
        error = self.solver.error[0,0]
        dis = self.solver.sol_dis_av[0,0]
        vec = self.solver.ori_sol_pd_coor[0,0,:]

        if ledu==0 or pdu==0:
            self.ans = self.ax.scatter(0,0,0,marker='x',color='k',s=10000)
            self.text_item = self.ax.text(-2.5,-2.5,-2, f'Usable LED:{ledu} \nUsable PD:{pdu}\nError:{error}')
            self.error_vec =self.ax.quiver (0,0,0,1,1,1,alpha=0,color = 'magenta')
        else:
            self.ans = self.ax.quiver(0,0,0,dis*vec[0],dis*vec[1],dis*vec[2],color='k')
            self.text_item = self.ax.text(-2.5,-2.5,-2, f'Usable LED:{ledu} \nUsable PD:{pdu}\nError:{error:.4E}')
            self.error_vec = self.quiver(dis*vec[0],dis*vec[1],dis*vec[2],self.testp.testp_pos[0,0]-dis*vec[0],self.testp.testp_pos[1,0]-dis*vec[1],self.env.testp.testp_pos[2,0]-dis*vec[2],color = 'magenta')


        # print('finished set up')
        # print(self.ax)
        # print(self.ax.collections)
        # print(self.sphere)
        # print(self.sphere.remove())
    
    def setup_sliders(self):
        """Setup the sliders for interactive parameter adjustment."""
        # Define slider parameters
        text = [
            r'$^{PL}t_x$', r'$^{PL}t_y$', r'$^{PL}t_z$',
            r'$Roll ^{PL}rx$', r'$Pitch ^{PL}ry$', r'$Yaw ^{PL}rz$',
            r'LED Count $L$', r'PD Count $P$',
            r'LED Lambertian Order $M\ell$', r'PD Lambertian Order $Mp$',
            r'Background Current $Ib (A)$', r'Bandwidth $B (Hz)$',
            r'LED Alpha Angle $^L\alpha (deg)$', r'PD Alpha Angle $^P\alpha (deg)$',
            r'PD Saturation Current $Is (A)$', r'Multipath Gain $Gm$'
        ]
        
        # Initial values
        init_val = np.concatenate([
            self.testp.testp_pos.flatten(),
            self.testp.testp_rot.flatten(),
            [self.led_system.num, self.pd_system.num,
             self.led_system.m, self.pd_system.m,
             np.log10(self.env.background), np.log10(self.env.bandwidth),
             np.rad2deg(self.led_system.ori_ang[0, 0]), np.rad2deg(self.pd_system.ori_ang[0, 0]),
             np.log10(self.pd_system.saturate), self.env.gain]
        ])
        
        # Slider ranges
        min_val = [
            -1.5, -1.5, 0, 0, 0, 0,
            3, 3, 1, 1, -6, 3, 0, 0, -6, 1
        ]
        max_val = [
            1.5, 1.5, 3, 2*np.pi, 2*np.pi, 2*np.pi,
            20, 20, 10, 10, -1, 12, 180, 180, 1, 2
        ]
        
        # Create sliders
        for i in range(len(min_val)):
            axamp = plt.axes([0.74, 0.8-(i*0.05), 0.12, 0.02])
            
            if 8 > i > 5:  # Integer sliders for LED and PD counts
                s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i], valstep=1)
            else:
                s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
            
            s.on_changed(self.on_slider_changed)
            self.sliders.append(s)
        
        # Format display values for logarithmic sliders
        self.sliders[11].valtext.set_text(f'{self.env.bandwidth:.4E}')
        self.sliders[10].valtext.set_text(f'{self.env.background:.4E}')
        self.sliders[14].valtext.set_text(f'{self.pd_system.saturate:.4E}')
        
        # # Add reset button
        # reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        # self.reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')
        # self.reset_button.on_clicked(self.reset)
    
    # def reset(self, event):
    #     """Reset sliders to initial values."""
    #     for slider in self.sliders:
    #         slider.reset()
    
    def on_slider_changed(self, val):
        """
        Handler for slider value changes.
        
        Args:
            val: New slider value (not used directly, we read all sliders)
        """
        # Update test point position and rotation
        self.testp.testp_pos = np.array([[
            self.sliders[0].val,
            self.sliders[1].val,
            self.sliders[2].val
        ]]).T
        
        self.testp.testp_rot = np.array([[
            self.sliders[3].val,
            self.sliders[4].val,
            self.sliders[5].val
        ]]).T
        
        # Update hardware parameters
        self.led_system.num = int(self.sliders[6].val)
        self.pd_system.num = int(self.sliders[7].val)
        self.led_system.m = self.sliders[8].val
        self.pd_system.m = self.sliders[9].val

        
        # Update solver parameters
        self.env.background = 10 ** self.sliders[10].val
        self.env.bandwidth = 10 ** self.sliders[11].val
        
        # Update hardware orientation
        led_alpha = np.deg2rad(self.sliders[12].val)
        pd_alpha = np.deg2rad(self.sliders[13].val)
        self.led_system.set_config(self.led_system.config_num, led_alpha)
        self.pd_system.set_config(self.pd_system.config_num, pd_alpha)
        
        # Update saturation and gain
        self.pd_system.saturate = 10 ** self.sliders[14].val
        self.env.gain = self.sliders[15].val
        
        # Format display values for logarithmic sliders
        self.sliders[11].valtext.set_text(f'{self.env.bandwidth:.4E}')
        self.sliders[10].valtext.set_text(f'{self.env.background:.4E}')
        self.sliders[14].valtext.set_text(f'{self.pd_system.saturate:.4E}')
        
        # Update visualization
        self.solve_and_update()
    
    def solve_and_update(self):
        """Solve positioning problem and update visualization."""
        # Remove old elements

        self.sphere.remove()
        self.axis_item.remove()
        self.ans.remove()
        self.error_vec.remove()
        
        for text in self.led_text:
            if text:
                text.remove()
        
        self.env.simulate_pd_sig()
        self.solver.solve_mulmul()


        testp_pos = self.testp.testp_pos
        testp_rot = self.testp.testp_rot

        # Get rotation matrix for coordinate system visualization
        arrow = 0.5 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
        arrow_rot = self.env.rotate_mat(testp_rot.flatten()) @ arrow
        
        # Update sphere position
        u, v = np.meshgrid(np.linspace(0, 2*np.pi, 20), np.linspace(0, np.pi, 20))
        x = 0.5 * np.cos(u) * np.sin(v)
        y = 0.5 * np.sin(u) * np.sin(v)
        z = 0.5 * np.cos(v)
        self.sphere = self.ax.plot_wireframe(
            x + testp_pos[0, 0],
            y + testp_pos[1, 0],
            z + testp_pos[2, 0],
            color="w", alpha=0.2, edgecolor="#808080"
        )
        
        # Update coordinate system visualization
        self.axis_item = self.ax.quiver(
            testp_pos[0, 0], testp_pos[1, 0], testp_pos[2, 0],
            arrow_rot[0, :], arrow_rot[1, :], arrow_rot[2, :],
            arrow_length_ratio=0.1, color='b'
        )
        
        # Update coordinate labels
        self.led_text[0] = self.ax.text(
            testp_pos[0, 0] + 1.1 * arrow_rot[0, 0],
            testp_pos[1, 0] + 1.1 * arrow_rot[1, 0],
            testp_pos[2, 0] + 1.1 * arrow_rot[2, 0],
            'x', color='b'
        )
        self.led_text[1] = self.ax.text(
            testp_pos[0, 0] + 1.1 * arrow_rot[0, 1],
            testp_pos[1, 0] + 1.1 * arrow_rot[1, 1],
            testp_pos[2, 0] + 1.1 * arrow_rot[2, 1],
            'y', color='b'
        )
        self.led_text[2] = self.ax.text(
            testp_pos[0, 0] + 1.1 * arrow_rot[0, 2],
            testp_pos[1, 0] + 1.1 * arrow_rot[1, 2],
            testp_pos[2, 0] + 1.1 * arrow_rot[2, 2],
            'z', color='b'
        )
        
        
        # Extract results
        pdu = self.solver.pdu[0,0]
        ledu = self.solver.ledu[0,0]
        error = self.solver.error[0,0]
        dis = self.solver.sol_dis_av[0,0]
        vec = self.solver.ori_sol_pd_coor[0,0,:]

        if ledu == 0 or pdu == 0:
            self.ans = self.ax.scatter(0, 0, 0, marker='x', color='k', s=10000)
            self.text_item.set_text(f'Usable LED: {ledu} \nUsable PD: {pdu}\nError: -')
            self.error_vec = self.ax.quiver(0, 0, 0, 1, 1, 1, alpha=0, color='magenta')
        else:
            self.ans = self.ax.quiver(0, 0, 0, dis*vec[0], dis*vec[1], dis*vec[2], color='k')
            self.text_item.set_text(f'Usable LED: {ledu} \nUsable PD: {pdu}\nError: {error:.4E}')
            self.error_vec = self.ax.quiver(
                dis*vec[0], dis*vec[1], dis*vec[2],
                testp_pos[0, 0]-dis*vec[0],
                testp_pos[1, 0]-dis*vec[1],
                testp_pos[2, 0]-dis*vec[2],
                color='magenta'
            )
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Show the interactive visualization."""
        plt.show()


class InteractiveMultiVisualizer:
    """
    Interactive figure of preset scenario(test points) with adjustable configurations, 
    visualizing the performance of the positioning system.

    """
    
    def __init__(self, led_system: LEDSystem, pd_system: PDSystem, 
                 solver: Solver, scenario:int=2, space_size:int=10, rot_max:int=180):
        """
        Initialize the interactive visualizer.
        
        Args:
            led_system: LED system instance
            pd_system: PD system instance
            solver: Positioning solver instance
            scenario: Scenario number
            space_size: Size of space for scenario 3
            rot_max: Maximum rotation angle for polar plots
        """
        self.led_system = led_system
        self.pd_system = pd_system
        self.solver = solver
        self.env = solver.env
        self.scenario = scenario
        self.space_size = space_size
        self.rot_max = rot_max
        self.test_point = self.env.testp
        
        # Track maximum performance for optimization
        self.max_count = 0
        self.max_led_alpha = 0
        self.max_pd_alpha = 0
        
        # Initialize figure and axes
        self.fig = plt.figure(figsize=(15, 8))
        self.setup_plot()
        
        # Initialize sliders and widgets
        self.sliders = []
        self.setup_sliders()
        
        # Solve initial configuration and update plot
        self.generate_collections()
    
    def setup_plot(self):
        """Setup the plots for visualization."""
        self.fig.subplots_adjust(wspace=0.3, hspace=0.3)
        
        # Plot for translation samples with color indicating success rate
        self.ax1 = self.fig.add_subplot(2, 4, 1, projection='3d')
        self.ax1.set_box_aspect(aspect=(1, 1, 1))
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.ax1.set_zlabel('z')
        self.ax1.grid(True)
        
        # Set axis limits based on scenario
        if self.scenario == 2:
            self.ax1.set_xlim3d(-3, 3)
            self.ax1.set_ylim3d(-3, 3)
            self.ax1.set_zlim3d(-3, 3)
        elif self.scenario == 3:
            self.ax1.set_xlim3d(-self.space_size/2, self.space_size/2)
            self.ax1.set_ylim3d(-self.space_size/2, self.space_size/2)
            self.ax1.set_zlim3d(0, self.space_size)
        else:
            self.ax1.set_xlim3d(-1.5, 1.5)
            self.ax1.set_ylim3d(-1.5, 1.5)
            self.ax1.set_zlim3d(0, 3)
        # self.sc1 = self.ax1.scatter(
        #     self.test_point.testp_pos[0,:],
        #     self.test_point.testp_pos[1,:],
        #     self.test_point.testp_pos[2,:],
        #     c = count_kpos,cmap=colormap,norm = normalizep,alpha=0.5)
        # self.ax1.scatter(0,0,0,color='k',marker='x')
        
        self.ax1.set_title('Translation Samples')
        
        # Plot for rotation samples
        self.ax2 = self.fig.add_subplot(2, 5, 6, projection='polar')
        self.ax2.set_title('Rotation Samples')
        self.ax2.text(1, 1, 'pitch (degrees)', rotation=15)
        self.ax2.text(np.deg2rad(60), 80, 'yaw (degrees)')
        self.ax2.set_ylim([0, self.rot_max])
        
        # Plot for effective translation range
        self.ax3 = self.fig.add_subplot(2, 5, 2, projection='3d')
        self.ax3.set_box_aspect(aspect=(1, 1, 1))
        self.ax3.set_xlabel('x')
        self.ax3.set_ylabel('y')
        self.ax3.set_zlabel('z')
        self.ax3.grid(True)
        
        # Set axis limits based on scenario
        if self.scenario == 2:
            self.ax3.set_xlim3d(-3, 3)
            self.ax3.set_ylim3d(-3, 3)
            self.ax3.set_zlim3d(-3, 3)
        else:
            self.ax3.set_xlim3d(-1.5, 1.5)
            self.ax3.set_ylim3d(-1.5, 1.5)
            self.ax3.set_zlim3d(0, 3)
        
        self.ax3.set_title('Effective Translation Range')
        
        # Plot for effective rotation range
        self.ax4 = self.fig.add_subplot(2, 5, 7, projection='polar')
        self.ax4.set_title('Effective Rotation Range')
        self.ax4.text(0, 0, 'pitch (degrees)', rotation=15)
        self.ax4.text(np.deg2rad(60), 80, 'yaw (degrees)')
        self.ax4.set_ylim([0, self.rot_max])
        
        # Plot for LED configuration
        self.ax5 = self.fig.add_subplot(2, 5, 3, projection='3d')
        self.ax5.xaxis.set_ticklabels([])
        self.ax5.yaxis.set_ticklabels([])
        self.ax5.zaxis.set_ticklabels([])
        self.ax5.set_axis_off()
        self.ax5.set_box_aspect(aspect=(1, 1, 1))
        self.ax5.set_title('LED System')
        self.ax5.set_xlim3d(-1, 1)
        self.ax5.set_ylim3d(-1, 1)
        self.ax5.set_zlim3d(-1, 1)
        
        # Draw coordinate axes and x-y plane for LED plot
        self.ax5.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color='grey')
        self.ax5.text(1.1, 0, 0, 'x', color='grey')
        self.ax5.text(0, 1.1, 0, 'y', color='grey')
        self.ax5.text(0, 0, 1.1, 'z', color='grey')
        
        # Draw x-y plane
        a = np.linspace(0, 2*np.pi, 20)
        b = np.linspace(0, 1, 10)
        r = np.outer(b, np.cos(a))
        o = np.outer(b, np.sin(a))
        zeror = np.zeros(r.shape)
        self.ax5.plot_surface(r, o, zeror, color="grey", alpha=0.25)
        
        # Draw unit sphere
        u, v = np.meshgrid(np.linspace(0, 2*np.pi, 20), np.linspace(0, np.pi, 21))
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        self.ax5.plot_wireframe(x, y, z, color="w", alpha=0.15, edgecolor="#808080")
        
        # Plot for PD configuration
        self.ax6 = self.fig.add_subplot(2, 5, 8, projection='3d')
        self.ax6.xaxis.set_ticklabels([])
        self.ax6.yaxis.set_ticklabels([])
        self.ax6.zaxis.set_ticklabels([])
        self.ax6.set_axis_off()
        self.ax6.set_box_aspect(aspect=(1, 1, 1))
        self.ax6.set_title('PD System')
        self.ax6.set_xlim3d(-1, 1)
        self.ax6.set_ylim3d(-1, 1)
        self.ax6.set_zlim3d(-1, 1)
        
        # Draw coordinate axes and x-y plane for PD plot
        self.ax6.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], color='grey')
        self.ax6.text(1.1, 0, 0, 'x', color='grey')
        self.ax6.text(0, 1.1, 0, 'y', color='grey')
        self.ax6.text(0, 0, 1.1, 'z', color='grey')
        self.ax6.plot_surface(r, o, zeror, color="grey", alpha=0.25)
        self.ax6.plot_wireframe(x, y, z, color="w", alpha=0.15, edgecolor="#808080")
        
        # Add text for maximum performance
        self.max_text = self.fig.text(0.8, 0.1, 'Max: 0')

        self.sc1 = None
        self.sc2 = None
        self.sc3 = None
        self.sc4 = None
        self.sc5 = None
        self.sc6 = None
        
        # # Add reset button
        # reset_ax = plt.axes([0.76, 0.05, 0.1, 0.04])
        # self.reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')
        # self.reset_button.on_clicked(self.reset)
        
        # # Add save button
        # save_ax = plt.axes([0.88, 0.05, 0.1, 0.04])
        # self.save_button = Button(save_ax, 'Save', hovercolor='0.975')
        # self.save_button.on_clicked(self.save_results)
    
    def setup_sliders(self):
        """Setup the sliders for interactive parameter adjustment."""
        # Define slider parameters
        text = [
            r'Tolerance $To (m)$', r'Effective Ratio (%)',
            r'LED Count $L$', r'PD Count $P$',
            r'LED Lambertian Order $M\ell$', r'PD Lambertian Order $Mp$',
            r'Background Current $Ib (A)$', r'Bandwidth $B (Hz)$',
            r'LED Alpha Angle $^L\alpha (deg)$', r'PD Alpha Angle $^P\alpha (deg)$',
            r'PD Saturation Current $Is (A)$', r'Resistor $Rl (Ohm)$',
            r'Multipath Gain $Gm$'
        ]
        
        # Initial values
        init_val = np.array([
            self.solver.tolerance, self.solver.effective,
            self.led_system.num, self.pd_system.num,
            self.led_system.m, self.pd_system.m,
            np.log10(self.env.background), np.log10(self.env.bandwidth),
            np.rad2deg(self.led_system.ori_ang[0, 0]), np.rad2deg(self.pd_system.ori_ang[0, 0]),
            np.log10(self.pd_system.saturate), np.log10(self.pd_system.shunt),
            self.env.gain
        ])
        
        # Slider ranges
        min_val = [0, 0, 3, 3, 1, 1, -6, 3, 0, 0, -6, 3, 1]
        max_val = [0.5, 100, 20, 20, 10, 10, -3, 12, 180, 180, 1, 10, 3]
        
        # Create sliders
        for i in range(len(min_val)):
            axamp = plt.axes([0.76, 0.8-(i*0.05), 0.12, 0.02])
            
            if 4 > i > 1:  # Integer sliders for LED and PD counts
                s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i], valstep=1)
            else:
                s = Slider(axamp, text[i], min_val[i], max_val[i], valinit=init_val[i])
            
            s.on_changed(self.on_slider_changed)
            self.sliders.append(s)
        
        # Format display values for logarithmic sliders
        self.sliders[7].valtext.set_text(f'{self.env.bandwidth:.4E}')
        self.sliders[6].valtext.set_text(f'{self.env.background:.4E}')
        self.sliders[10].valtext.set_text(f'{self.pd_system.saturate:.4E}')
        self.sliders[11].valtext.set_text(f'{self.pd_system.shunt:.4E}')
        
        # # Add configuration selection
        # config_ax = plt.axes([0.76, 0.15, 0.2, 0.1])
        # self.config_radio = RadioButtons(
        #     config_ax, ('Config 0: Radial', 'Config 1: Radial+Center', 'Config 2: Dual-Ring'),
        #     active=0
        # )
        # self.config_radio.on_clicked(self.on_config_changed)
    
    # def reset(self, event):
    #     """Reset sliders to initial values."""
    #     for slider in self.sliders:
    #         slider.reset()
    
    # def save_results(self, event):
    #     """Save current results to file."""
    #     self.fig.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
        
    #     # Save parameters and results to text file
    #     with open('optimization_results.txt', 'w') as f:
    #         f.write(f"Optimization Results\n")
    #         f.write(f"==================\n")
    #         f.write(f"Date: {np.datetime64('today')}\n\n")
            
    #         f.write(f"Parameters:\n")
    #         f.write(f"Tolerance: {self.solver.tolerance:.4f} m\n")
    #         f.write(f"Effective Ratio: {self.solver.effective:.1f}%\n")
    #         f.write(f"LED Count: {self.led_system.num}\n")
    #         f.write(f"PD Count: {self.pd_system.num}\n")
    #         f.write(f"LED Lambertian Order: {self.led_system.m:.2f}\n")
    #         f.write(f"PD Lambertian Order: {self.pd_system.m:.2f}\n")
    #         f.write(f"Background Current: {self.env.background:.4E} A\n")
    #         f.write(f"Bandwidth: {self.env.bandwidth:.4E} Hz\n")
    #         f.write(f"LED Alpha Angle: {np.rad2deg(self.led_system.ori_ang[0, 0]):.1f} deg\n")
    #         f.write(f"PD Alpha Angle: {np.rad2deg(self.pd_system.ori_ang[0, 0]):.1f} deg\n")
    #         f.write(f"PD Saturation Current: {self.pd_system.saturate:.4E} A\n")
    #         f.write(f"Resistor: {self.pd_system.shunt:.4E} Ohm\n")
    #         f.write(f"Multipath Gain: {self.env.gain:.2f}\n\n")
            
    #         f.write(f"Results:\n")
    #         f.write(f"Total successful samples: {self.max_count}\n")
    #         f.write(f"Best LED Alpha: {self.max_led_alpha:.1f} deg\n")
    #         f.write(f"Best PD Alpha: {self.max_pd_alpha:.1f} deg\n")
        
    #     print(f"Results saved to optimization_results.png and optimization_results.txt")
    
    # def on_config_changed(self, label):
    #     """
    #     Handler for configuration radio button changes.
        
    #     Args:
    #         label: Selected configuration label
    #     """
    #     if label == 'Config 0: Radial':
    #         config_num = 0
    #     elif label == 'Config 1: Radial+Center':
    #         config_num = 1
    #     else:  # Config 2: Dual-Ring
    #         config_num = 2
        
    #     # Update hardware configuration
    #     led_alpha = np.deg2rad(self.sliders[8].val)
    #     pd_alpha = np.deg2rad(self.sliders[9].val)
    #     self.led_system.set_config(config_num, led_alpha)
    #     self.pd_system.set_config(config_num, pd_alpha)
        
    #     # Update visualization
    #     self.solve_and_update()
    
    def on_slider_changed(self, val):
        """
        Handler for slider value changes.
        
        Args:
            val: New slider value (not used directly, we read all sliders)
        """
        # Update solver parameters
        self.solver.tolerance = self.sliders[0].val
        self.solver.effective = self.sliders[1].val
        
        # Update hardware parameters
        self.led_system.num = int(self.sliders[2].val)
        self.pd_system.num = int(self.sliders[3].val)
        self.led_system.m = self.sliders[4].val
        self.pd_system.m = self.sliders[5].val
        
        # Update logarithmic parameters
        self.env.background = 10 ** self.sliders[6].val
        self.env.bandwidth = 10 ** self.sliders[7].val
        
        # Update hardware orientation
        led_alpha = np.deg2rad(self.sliders[8].val)
        pd_alpha = np.deg2rad(self.sliders[9].val)
        
        # # Get current configuration from radio button
        # config_label = self.config_radio.value_selected
        # if config_label == 'Config 0: Radial':
        #     config_num = 0
        # elif config_label == 'Config 1: Radial+Center':
        #     config_num = 1
        # else:  # Config 2: Dual-Ring
        #     config_num = 2
            
        self.led_system.set_config(self.led_system.config_num, led_alpha)
        self.pd_system.set_config(self.pd_system.config_num, pd_alpha)
        
        # Update other parameters
        self.pd_system.saturate = 10 ** self.sliders[10].val
        self.pd_system.shunt = 10 ** self.sliders[11].val
        self.env.gain = self.sliders[12].val
        
        # Format display values for logarithmic sliders
        self.sliders[7].valtext.set_text(f'{self.env.bandwidth:.4E}')
        self.sliders[6].valtext.set_text(f'{self.env.background:.4E}')
        self.sliders[10].valtext.set_text(f'{self.pd_system.saturate:.4E}')
        self.sliders[11].valtext.set_text(f'{self.pd_system.shunt:.4E}')
        
        self.env.simulate_pd_sig()
        self.solver.solve_mulmul()
        self.clear_collections()

        # Update visualization
        self.generate_collections()

    def clear_collections(self):
        # Remove old plots
        for sc in [self.sc1, self.sc2, self.sc3, self.sc4, self.sc5, self.sc6]:
            sc.remove()

    
    def generate_collections(self):
        """Solve positioning problem and update visualization."""
  
        # Solve positioning problem
        # self.solver.solve_mulmul()
        
        # Calculate counts and effective regions
        error = self.solver.error
        count_total = np.nansum(error < self.solver.tolerance)
        count_kpos = np.nansum(error < self.solver.tolerance, axis=1)
        count_krot = np.nansum(error < self.solver.tolerance, axis=0)
        effective_pos = count_kpos / self.test_point.krot >= self.solver.effective / 100
        effective_rot = count_krot / self.test_point.kpos >= self.solver.effective / 100
        
        # Update title
        self.fig.suptitle(f'Total samples within tolerance: {count_total}')
        
        # Create color mappings
        colormap = plt.cm.get_cmap('YlOrRd')
        normalizep = colors.Normalize(vmin=0, vmax=self.test_point.krot)
        normalizer = colors.Normalize(vmin=0, vmax=self.test_point.kpos)
        
        # Update translation samples plot
        self.sc1 = self.ax1.scatter(
            self.test_point.testp_pos[0, :],
            self.test_point.testp_pos[1, :],
            self.test_point.testp_pos[2, :],
            c=count_kpos,
            cmap=colormap,
            norm=normalizep,
            alpha=0.5
        )
        self.ax1.scatter(0, 0, 0, color='k', marker='x')
        
        # Update rotation samples plot
        self.sc2 = self.ax2.scatter(
            self.test_point.testp_rot[2, :],
            np.rad2deg(self.test_point.testp_rot[1, :]),
            c=count_krot,
            cmap=colormap,
            norm=normalizer
        )
        
        # Update effective translation range plot
        self.sc3 = self.ax3.scatter(
            self.test_point.testp_pos[0, effective_pos],
            self.test_point.testp_pos[1, effective_pos],
            self.test_point.testp_pos[2, effective_pos],
            color='b',
            alpha=0.5
        )
        self.ax3.scatter(0, 0, 0, color='k', marker='x')
        
        # Update effective rotation range plot
        self.sc4 = self.ax4.scatter(
            self.test_point.testp_rot[2, effective_rot],
            np.rad2deg(self.test_point.testp_rot[1, effective_rot]),
            color='b'
        )
        
        # Update LED configuration plot
        zeros = np.zeros((self.led_system.num,))
        self.sc5 = self.ax5.quiver(zeros, zeros, zeros,
                                  self.led_system.ori_car[0, :],
                                  self.led_system.ori_car[1, :],
                                  self.led_system.ori_car[2, :],
                                  color='b')
        
        # Update PD configuration plot
        zeros = np.zeros((self.pd_system.num,))
        self.sc6 = self.ax6.quiver(zeros, zeros, zeros,
                                  self.pd_system.ori_car[0, :],
                                  self.pd_system.ori_car[1, :],
                                  self.pd_system.ori_car[2, :],
                                  color='firebrick')
        
        # Update maximum performance tracking
        if count_total > self.max_count:
            self.max_count = count_total
            self.max_led_alpha = self.sliders[8].val
            self.max_pd_alpha = self.sliders[9].val
            self.max_text.set_text(f'Max: {self.max_count}\n'
                                  f'LED α: {self.max_led_alpha:.1f}°\n'
                                  f'PD α: {self.max_pd_alpha:.1f}°')
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Show the interactive visualization."""
        plt.show()


class ConfigOptimizer:
    """
    Optimizer for finding optimal system configuration.
    
    This class implements a grid search approach to find optimal parameter
    combinations for the positioning system.
    """
    
    def __init__(self, led_system: LEDSystem, pd_system: PDSystem, solver: Solver):
        """
        Initialize the optimizer.
        
        Args:
            led_system: LED system instance
            pd_system: PD system instance
            solver: Positioning solver instance
        """
        self.led_system = led_system
        self.pd_system = pd_system
        self.solver = solver
        self.env = solver.env
        self.best_params = {}
        self.best_score = 0
        
    def optimize(self, scenario: int = 2, space_size: int = 10, 
                 param_ranges: Optional[Dict] = None, save_results: bool = True):
        """
        Run optimization to find best parameter combination.
        
        Args:
            scenario: Scenario number
            space_size: Size of space for scenario 3
            param_ranges: Dictionary with parameter ranges for optimization
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with best parameters and score
        """
        # Default parameter ranges if not provided
        if param_ranges is None:
            param_ranges = {
                'led_alpha': np.linspace(10, 70, 7),  # degrees
                'pd_alpha': np.linspace(10, 70, 7),   # degrees
                'led_m': [1, 2, 5],                   # Lambertian orders
                'pd_m': [1, 2, 5],                    # Lambertian orders
                'config_num': [0, 1]                  # Configuration types
            }
        
        # Generate test points based on scenario
        test_point = TestPoint(scenario, {'ma': space_size})
        
        # Initialize best score and parameters
        self.best_score = 0
        self.best_params = {}
        
        # Setup progress tracking
        total_combinations = (len(param_ranges['led_alpha']) * 
                              len(param_ranges['pd_alpha']) * 
                              len(param_ranges['led_m']) * 
                              len(param_ranges['pd_m']) * 
                              len(param_ranges['config_num']))
        
        print(f"Starting optimization with {total_combinations} combinations...")
        progress = 0
        
        # Grid search through parameter combinations
        for config_num in param_ranges['config_num']:
            for led_m in param_ranges['led_m']:
                for pd_m in param_ranges['pd_m']:
                    self.led_system.lambertian_order = led_m
                    self.pd_system.lambertian_order = pd_m
                    
                    for led_alpha_deg in param_ranges['led_alpha']:
                        for pd_alpha_deg in param_ranges['pd_alpha']:
                            # Update configuration
                            led_alpha = np.deg2rad(led_alpha_deg)
                            pd_alpha = np.deg2rad(pd_alpha_deg)
                            self.led_system.set_config(config_num, led_alpha)
                            self.pd_system.set_config(config_num, pd_alpha)
                            
                            # Solve positioning problem
                            results = self.solver.solve(test_point.testp_pos, test_point.testp_rot)
                            
                            # Calculate score (total samples within tolerance)
                            error = results['error']
                            count_total = np.nansum(error < self.solver.tolerance)
                            
                            # Check if this is the best score so far
                            if count_total > self.best_score:
                                self.best_score = count_total
                                self.best_params = {
                                    'config_num': config_num,
                                    'led_alpha': led_alpha_deg,
                                    'pd_alpha': pd_alpha_deg,
                                    'led_m': led_m,
                                    'pd_m': pd_m,
                                    'score': count_total
                                }
                            
                            # Update progress
                            progress += 1
                            if progress % 10 == 0:
                                print(f"Progress: {progress}/{total_combinations} "
                                      f"({progress/total_combinations*100:.1f}%)")
        
        # Print results
        print("\nOptimization completed!")
        print(f"Best score: {self.best_score}")
        print(f"Best parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        # Save results if requested
        if save_results:
            self._save_results()
        
        return self.best_params
    
    def _save_results(self):
        """Save optimization results to file."""
        # Save parameters and results to text file
        with open('optimization_results.txt', 'w') as f:
            f.write(f"Optimization Results\n")
            f.write(f"==================\n")
            f.write(f"Date: {np.datetime64('today')}\n\n")
            
            f.write(f"Best Parameters:\n")
            for key, value in self.best_params.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nSolver Configuration:\n")
            f.write(f"Tolerance: {self.solver.tolerance:.4f} m\n")
            f.write(f"Effective Ratio: {self.solver.effective:.1f}%\n")
            f.write(f"Background Current: {self.env.background:.4E} A\n")
            f.write(f"Bandwidth: {self.env.bandwidth:.4E} Hz\n")
            f.write(f"PD Saturation Current: {self.pd_system.saturate:.4E} A\n")
            f.write(f"Shunt Resistance: {self.pd_system.shunt:.4E} Ohm\n")
            f.write(f"Multipath Gain: {self.env.gain:.2f}\n")
        
        print(f"Results saved to optimization_results.txt")