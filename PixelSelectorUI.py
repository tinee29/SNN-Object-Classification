import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import os
from utils import get_hand_mask_548

class PixelSelectorUI:
    def __init__(self):
        """Interactive 32x32 mask painter for selecting valid hand taxels."""
        self.output_dir = "masks"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.valid_mask = get_hand_mask_548()
        self.selection_mask = np.zeros(1024, dtype=np.int8)
        self.is_drawing = False 
        self.target_val = 1
        
        # 0=invalid, 1=valid, 2=selected
        self.display_data = np.zeros(1024, dtype=int)
        self.display_data[self.valid_mask] = 1 
        
        self.setup_plot()
        self.setup_widgets()
        
        ui = widgets.VBox([
            self.status_lbl,
            self.fig.canvas,
            widgets.HBox([self.name_entry, self.clear_btn, self.save_btn]),
            self.log_out
        ])
        
        display(ui)

    def setup_plot(self):
        """Initialize the grid plot and mouse event bindings."""
        plt.close('all')
        plt.ioff() 
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        plt.ion()

        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.resizable = False
        
        self.cmap = mcolors.ListedColormap(['#eeeeee', '#90EE90', '#ef5350'])
        self.norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], self.cmap.N)
        
        self.im = self.ax.imshow(
            self.display_data.reshape(32, 32), 
            cmap=self.cmap, norm=self.norm, origin='upper'
        )
        
        self.ax.set_xticks(np.arange(-.5, 32, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, 32, 1), minor=True)
        self.ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
        self.ax.tick_params(which='minor', size=0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Click & Drag to Paint (Click again to Erase)", fontsize=10)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)

    def setup_widgets(self):
        """Create controls for status, naming, clearing, and saving."""
        self.status_lbl = widgets.Label(f"Selected: 0 / {np.sum(self.valid_mask)}")
        self.name_entry = widgets.Text(value='custom_mask', placeholder='Mask Name', layout=widgets.Layout(width='200px'))
        self.clear_btn = widgets.Button(description='Clear', button_style='warning', icon='refresh', layout=widgets.Layout(width='100px'))
        self.save_btn = widgets.Button(description='Save', button_style='success', icon='check', layout=widgets.Layout(width='100px'))
        self.log_out = widgets.Output()
        
        self.clear_btn.on_click(self.clear_selection)
        self.save_btn.on_click(self.save_mask)

    def update_view(self):
        """Refresh the image and selection counter."""
        self.display_data[:] = 0
        self.display_data[self.valid_mask] = 1
        self.display_data[self.selection_mask == 1] = 2
        
        self.im.set_data(self.display_data.reshape(32, 32))
        self.fig.canvas.draw_idle()
        self.status_lbl.value = f"Selected: {np.sum(self.selection_mask)} / {np.sum(self.valid_mask)}"

    def paint_pixel(self, event, target_val):
        """Set one pixel from a mouse event if it is valid and changed."""
        if event.inaxes != self.ax: return
        
        c = int(event.xdata + 0.5)
        r = int(event.ydata + 0.5)
        
        if 0 <= r < 32 and 0 <= c < 32:
            idx = r * 32 + c
            if self.valid_mask[idx] and self.selection_mask[idx] != target_val:
                self.selection_mask[idx] = target_val
                self.update_view()

    def on_click(self, event):
        """Start a stroke and decide whether to paint or erase."""
        if event.inaxes != self.ax: return
        self.is_drawing = True
        
        c = int(event.xdata + 0.5)
        r = int(event.ydata + 0.5)
        idx = r * 32 + c
        
        if 0 <= idx < 1024 and self.valid_mask[idx]:
            self.target_val = 1 - self.selection_mask[idx]
            self.paint_pixel(event, self.target_val)

    def on_release(self, event):
        """Stop the current stroke."""
        self.is_drawing = False

    def on_drag(self, event):
        """Continue painting while dragging with mouse pressed."""
        if self.is_drawing:
            self.paint_pixel(event, self.target_val)

    def clear_selection(self, b):
        """Clear all selected pixels."""
        self.selection_mask[:] = 0
        self.update_view()

    def save_mask(self, b):
        """Save the current selection mask as a `.npy` file."""
        with self.log_out:
            clear_output()
            name = self.name_entry.value.strip()
            if not name:
                print("❌ Please enter a name.")
                return
            if not name.endswith(".npy"): name += ".npy"
            
            path = os.path.join(self.output_dir, name)
            np.save(path, self.selection_mask)
            print(f"✅ Saved to: {path}")
            print(f"🔲 Active Pixels: {np.sum(self.selection_mask)}")