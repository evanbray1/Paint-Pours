# Paint Pour Digital Art Generator

![Sample Paint Pour Output](PLACEHOLDER_FOR_MAIN_IMAGE)

## Overview
This project digitally replicates the visual style of "paint pour" artwork, where an artist pours multiple colors of acrylic paint onto a canvas to create beautiful, organic patterns. This package attempts to recreate this style of art by thinking of it as a carefully-crafted 3D surface, shown as a filled contour plot. The program generates aesthetic, filled-contour images using fractal/Perlin noise, custom segmented colormaps, and optional Voronoi-style cell overlays, resulting in images that closely resemble real paint pours.

## Intended Audience
This tool is presented here for categories of people. 
1. Anyone interested in creating wallpapers, backgrounds, or banner art for various digital mediums. The code is written to be readable and accessible to true Python novices, making it a great starting point for beginners who want to experiment with generative art.
2. People who requested a code sample, for professional purposes. 

## Features
- **Automatic Generation:** Easily create paint pour-style images with a single function call.
- **Customizable Parameters:** Over a dozen variables can be tweaked manually for creative control, or left to be randomly chosen from sensible ranges.
- **Prominent Cells Mode:** The `prominent_cells` option produces a fundamentally different style of image by manipulating variable ranges, resulting in large "cells" (as the paint pour artists call them) being prominently featured in the foreground. This style was popular enough during user-testing (aka sharing photos with friends) that it warranted its own dedicated preset.
- **Intermediate Plots:** Optionally display detailed intermediate plots to illustrate the many steps of the image generation process.
- **Voronoi Cell Overlays:** Add cell-like structures to simulate effects seen in real acrylic pours. Although best visualized with the 'prominent_cells' argument, more features are coming to incorporate these cells structures at smaller scales in ways that more closely mimic real paint pours.
- **Flexible Saving:** Save images to a specified directory, or simply display them interactively.

![Intermediate Plot Example](PLACEHOLDER_FOR_INTERMEDIATE_IMAGE)

## Example Usage
A minimal example to generate a paint pour image:

```python
import paint_pour_tools as pptools

pptools.generate_paint_pour_images(
    image_dimensions=[1920, 1080],
    num_images=1,
    display_final_image=True,
    save_image=True,
    prominent_cells=True
)
```

For more advanced usage, you can specify additional parameters to customize the output. See the docstrings for the `generate_paint_pour_image()` function.

## Getting Started
1. Install the required Python packages (see below).
2. Run `Paint Pouring Sandbox.py`.
3. Adjust parameters as desired to experiment with different styles.

## Installation
This project requires no special installation instructions, aside from several common modules like NumPy, Matplotlib, and Scipy. 

## Contact
Questions, suggestions, or feedback? Contact the author:
**Evan Bray** — Bray.EvanP@gmail.com

## Image Gallery
Sample outputs and intermediate plots will be added here soon.

![Gallery Placeholder](PLACEHOLDER_FOR_GALLERY)
