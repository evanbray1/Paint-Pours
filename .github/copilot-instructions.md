# Copilot Instructions for Paint-Pours

## Project Overview
This repository generates digital art in the style of acrylic paint pours using Python. The main logic is in `paint_pour_tools.py`, with a sandbox script for experimentation. Images are created using fractal/Perlin noise, custom colormaps, and optional Voronoi overlays.

## Key Files
- `paint_pour_tools.py`: Core image generation functions. Main entry: `generate_paint_pour_image()` and `generate_paint_pour_images()`.
- `Paint Pouring sandbox.py`: Example usage and experimentation. Use this for quick tests and parameter tweaks.
- `README.md`: Contains usage examples, feature descriptions, and advanced tips.

## Developer Workflows
- **Run Art Generation**: Execute `Paint Pouring sandbox.py` to generate images. Adjust parameters in the script for different styles.
- **Interactive Plots**: Use `plt.close('all')` and `plt.pause(0.1)` between images to ensure proper plot window handling in VS Code.
- **Parameter Tweaking**: Most function arguments have sensible defaults and can be randomized. For custom results, pass explicit values (see README and docstrings).
- **Saving/Displaying**: Images can be saved to disk or displayed interactively. Output directory is configurable.

## Patterns & Conventions
- **Custom Colormaps**: Pass a list of hex colors to `custom_cmap_colors` for unique palettes.
- **Prominent Cells Mode**: Set `prominent_cells=True` for images with large cell-like structures. This changes several internal parameters for a distinct look.
- **Intermediate Plots**: Enable `show_intermediate_plots=True` for step-by-step visual debugging.
- **Randomization**: If parameters are omitted, the code intelligently randomizes them within reasonable bounds.
- **Reproducibility**: Use the `seed` argument to reproduce results.

## External Dependencies
- Requires: `numpy`, `matplotlib`, `scipy`, `cv2` (OpenCV)
- No build system or test suite is present; run scripts directly in Python 3.6+.

## Integration Points
- All image generation is handled via `paint_pour_tools.py`. No external APIs or services are used.
- Output is local PNG files and interactive matplotlib windows.

## Example Usage
```python
import paint_pour_tools as pptools
pptools.generate_paint_pour_images(image_dimensions=[1920, 1080], prominent_cells=True)
```

## Tips for AI Agents
- Reference the README and function docstrings for parameter details and advanced usage.
- When adding features, follow the pattern of sensible defaults and optional randomization.
- Keep code readable for Python beginners; avoid unnecessary complexity.
- Use the sandbox script for new experiments and demonstrations.

---
If any conventions or workflows are unclear, ask the user for clarification or examples from their recent work.
