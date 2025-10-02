# Copilot Instructions for Paint-Pours

## Project Overview
This repository generates digital art in the style of acrylic paint pours using Python. The main logic is in `paint_pour_tools.py`, with a sandbox script for experimentation. Images are created using fractal/Perlin noise, custom colormaps, and optional Voronoi overlays. The project now includes metadata-based image generation for creating similar images from previously saved parameters.

## Key Files
- `paint_pour_tools.py`: Core image generation functions. Main entry points: `generate_paint_pour_image()`, `generate_paint_pour_images()`, and `create_similar_images()`.
- `Paint Pouring sandbox.py`: Example usage and experimentation. Use this for quick tests and parameter tweaks.
- `README.md`: Contains usage examples, feature descriptions, and advanced tips.

## Developer Workflows
- **Run Art Generation**: Execute `Paint Pouring sandbox.py` to generate images. Adjust parameters in the script for different styles.
- **Metadata-Based Generation**: Use metadata CSV files from previous generations to create similar images with `create_similar_images()`.
- **Interactive Plots**: Use `plt.close('all')` and `plt.pause(0.1)` between images to ensure proper plot window handling in VS Code.
- **Parameter Tweaking**: Most function arguments have sensible defaults and can be randomized. For custom results, pass explicit values (see README and docstrings).
- **Saving/Displaying**: Images can be saved to disk or displayed interactively. Output directory is configurable.

## Patterns & Conventions
- **Custom Colormaps**: Pass a list of hex colors to `custom_cmap_colors` for unique palettes.
- **Prominent Cells Mode**: Set `prominent_cells=True` for images with large cell-like structures. This changes several internal parameters for a distinct look.
- **Metadata Files**: Each generated image automatically saves a CSV metadata file containing all parameters. These can be used with `create_similar_images()` to recreate similar styles.
- **Parameter Overrides**: The `create_similar_images()` function allows user overrides of metadata parameters (e.g., different resolution, colormap levels).
- **Intermediate Plots**: Enable `show_intermediate_plots=True` for step-by-step visual debugging (automatically disabled in `create_similar_images()` unless overridden).
- **Randomization**: If parameters are omitted, the code intelligently randomizes them within reasonable bounds.
- **Reproducibility**: Use the `seed` argument to reproduce results. The `create_similar_images()` function generates unique seeds for each similar image.

## External Dependencies
- Requires: `numpy`, `matplotlib`, `scipy`, `cv2` (OpenCV), `numba`
- No build system or test suite is present; run scripts directly in Python 3.6+.

## Integration Points
- All image generation is handled via `paint_pour_tools.py`. No external APIs or services are used.
- Output is local PNG files and interactive matplotlib windows.
- Metadata files are CSV format with 'attribute' and 'value' columns.

## Example Usage
```python
import paint_pour_tools as pptools

# Generate single or multiple images
pptools.generate_paint_pour_images(image_dimensions=[1920, 1080], prominent_cells=True)

# Create similar images from metadata
pptools.create_similar_images('path/to/metadata.csv', num_images=5)

# Override specific parameters from metadata
pptools.create_similar_images(
    'path/to/metadata.csv', 
    num_images=3,
    image_dimensions=[2560, 1440],
    num_colormap_levels=50
)
```

## Tips for AI Agents
- Reference the README and function docstrings for parameter details and advanced usage.
- When adding features, follow the pattern of sensible defaults and optional randomization.
- Keep code readable for Python beginners; avoid unnecessary complexity.
- Use the sandbox script for new experiments and demonstrations.
- The `create_similar_images()` function handles data type conversion from CSV strings back to appropriate Python types.
- Always create the 'from_metadata_file' subdirectory for similar images to keep outputs organized.

---
If any conventions or workflows are unclear, ask the user for clarification or examples from their recent work.
