import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SmartHandAverager:
    def __init__(self, finger_kernel=(3, 2), palm_kernel=(2, 2), thumb_kernel=(2, 2)):
        """Build a hand-region averager over a 32x32 tactile grid.

        Args:
            finger_kernel (tuple): (height, width) used for finger super-pixels.
            palm_kernel (tuple): (height, width) used for palm super-pixels.
            thumb_kernel (tuple): (height, width) used for thumb super-pixels.
        """
        self.finger_h, self.finger_w = finger_kernel
        self.palm_h, self.palm_w = palm_kernel
        self.thumb_h, self.thumb_w = thumb_kernel

        self.groups = []
        self.centers = []
        self.weights = []
        self._build_mapping()
        
    def _add_group(self, rows, cols):
        """Add one super-pixel group from row/column ranges."""
        r_grid, c_grid = np.meshgrid(rows, cols, indexing='ij')
        
        # Flatten in row-major order for 32x32 indexing.
        indices = (r_grid * 32 + c_grid).flatten()
        
        self.groups.append(indices)
        self.centers.append((np.mean(rows), np.mean(cols)))
        
        # Uniform averaging weights for this group.
        w = np.zeros(1024)
        w[indices] = 1.0 / len(indices)
        self.weights.append(w)

    def _build_mapping(self):
        """Create region groups for fingers, palm, and thumb."""
        
        finger_block_starts = [0, 6, 10, 15] 
        
        for r_start in finger_block_starts:
            # Tile each finger block (3 rows high) with the configured kernel.
            for r in range(r_start, r_start + 3, self.finger_h):
                r_end = min(r + self.finger_h, r_start + 3)
                if r >= r_end: continue

                for c in range(0, 14, self.finger_w):
                    c_end = min(c + self.finger_w, 14)
                    if c >= c_end: continue
                    
                    rows = range(r, r_end)
                    cols = range(c, c_end)
                    self._add_group(rows, cols)

        # Palm region: upper-right area.
        for r in range(0, 18, self.palm_h):
            r_end = min(r + self.palm_h, 18)
            
            for c in range(14, 32, self.palm_w):
                c_end = min(c + self.palm_w, 32)
                
                rows = range(r, r_end)
                cols = range(c, c_end)
                self._add_group(rows, cols)

        # Thumb region: lower-right strip.
        for r in range(18, 32, self.thumb_h):
            r_end = min(r + self.thumb_h, 32)
            
            for c in range(25, 29, self.thumb_w):
                c_end = min(c + self.thumb_w, 29)
                
                rows = range(r, r_end)
                cols = range(c, c_end)
                self._add_group(rows, cols)
                
        # Shape: (1024, num_groups), used by x @ matrix.
        self.matrix = np.array(self.weights).T.astype(np.float32)
        print(f"Built Averager: {len(self.groups)} Super-Pixels created.")

    def apply(self, x):
        """Apply grouped averaging.

        Args:
            x: Input with trailing shape `(1024,)` or `(32, 32)`.

        Returns:
            Averaged features with trailing dimension `num_groups`.
        """
        original_shape = x.shape
        if x.shape[-1] != 1024:
            x = x.reshape(*original_shape[:-2], 1024)
            
        x_avg = x @ self.matrix
        
        return x_avg

    def plot_layout(self, title=None, save_path=None):
        """Plot the super-pixel layout used by the averager.

        Args:
            title (str | None): Figure title. Defaults to group-count title.
            save_path (str | None): Optional filename stem under `figures/`.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        # Match the styling used by mask plots.
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams.update({
            'figure.dpi': 400,
            'axes.grid': False,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': False,
            'axes.spines.bottom': False,
        })

        fig, ax = plt.subplots(figsize=(6.0, 6.0))

        ax.set_facecolor('white')

        # Limits for a 32x32 rotated grid.
        ax.set_xlim(-0.5, 31.5)
        ax.set_ylim(-0.5, 31.5)

        # Light cell grid for readability.
        ax.set_xticks(np.arange(-0.5, 32, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 32, 1), minor=True)
        ax.grid(which='minor', color='#e0e0e0', linestyle='-', linewidth=0.4, alpha=0.7)
        ax.tick_params(which='minor', size=0)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Draw one rectangle per group.
        for indices in self.groups:
            rows = indices // 32
            cols = indices % 32

            r_min_orig, r_max_orig = rows.min(), rows.max() + 1
            c_min_orig, c_max_orig = cols.min(), cols.max() + 1

            color = "#48bdeb"
            if r_max_orig <= 18 and c_max_orig <= 14:
                color = "#ff4e4e"
            if r_min_orig >= 18:
                color = "#59e159"

            # Rotate clockwise to match the display convention.
            rows_rot = 31 - cols
            cols_rot = 31 - rows

            r_min = rows_rot.min()
            r_max = rows_rot.max() + 1
            c_min = cols_rot.min()
            c_max = cols_rot.max() + 1

            rect = patches.Rectangle(
                (c_min - 0.5, r_min - 0.5),
                c_max - c_min, r_max - r_min,
                linewidth=0.8,
                edgecolor='white',
                facecolor=color,
                alpha=0.95
            )
            ax.add_patch(rect)

        if title is None:
            title = f"Averaging Layout ({len(self.groups)} inputs)"
        ax.set_title(title, fontsize=24, fontweight='semibold', pad=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(f'figures/{save_path}.png', dpi=400, bbox_inches='tight', facecolor='white')

        plt.show()