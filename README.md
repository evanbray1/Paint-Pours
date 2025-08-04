# Paint Pour Digital Art Generator

<p align="center"><img width="652" height="1102" alt="image" src="https://github.com/user-attachments/assets/8257179e-fcb8-42fc-92e0-cfd89093e73f" /></p>



## Overview
This project seeks to digitally replicate the visual style of "paint pour" artwork, where an artist pours multiple colors of acrylic paint onto a canvas to create beautiful, organic patterns. This package attempts to recreate this style of art by thinking of it as a carefully-crafted 3D surface, shown as a filled contour plot. The program generates aesthetic, filled-contour images using fractal/Perlin noise, custom segmented colormaps, and optional Voronoi-style cell overlays, resulting in images that closely resemble real paint pours.

## Intended Audience
This tool is presented here for categories of people. 
1. Anyone interested in creating wallpapers, backgrounds, or banner art for various digital mediums. The code is written to be readable and accessible to true Python novices, making it a great starting point for beginners who want to experiment with generative art.
2. People who requested a code sample, for professional purposes. 

## Features
- **Automatic Generation:** Easily create paint pour-style images with a single function call.
- **Customizable Parameters:** Over a dozen variables can be tweaked manually for creative control, or left to be randomly chosen from sensible ranges.
- **Voronoi Cell Overlays:** Add cell-like structures to simulate effects seen in real acrylic pours. Although best visualized with the 'prominent_cells' argument, more features are coming to incorporate these cells structures at smaller scales in ways that more closely mimic real paint pours.
- **Prominent Cells Mode:** The `prominent_cells` option produces a fundamentally different style of image by manipulating variable ranges, resulting in large "cells" (as the paint pour artists call them) being prominently featured in the foreground. This style was popular enough during user-testing (aka sharing photos with friends) that it warranted its own dedicated preset.
- **Flexible Saving:** Save images to a specified directory, already properly rendered in the desired resolution, or simply display them interactively.
- **Intermediate plots for troubleshooting:** Optionally display detailed intermediate plots to illustrate the many steps of the image generation process. A selection of these plots are shown below. 

<img width="2012" height="1131" alt="image" src="https://github.com/user-attachments/assets/cd36aaa1-a248-4df1-bcf6-6b0165aef093" />


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

### Features I would like to add someday
- **Silicone oil "cells":** When silicone oil is added to the paint mixture in certain layers, it changers the surface tension properties of that layer, which results in that layer breaking when it gets thin enough, and allowing the lower layers to show through. For example, the below image has lots of these cells. This has proven a tricky feature to simulate.
  - ![paint pour with cells](https://github.com/user-attachments/assets/db4a4a1e-d9e6-44b5-8413-62084241b013)

- **Inter-layer blending:** Sometimes, the artist will purposefully, loosely mix two colors of paint so they are not quite homogenous _prior_ to pouring them onto the canvas. The image shown at the top of this ReadMe exhibits some of this. This can be done by craftily converting some portions of the segmented colormap to a continuous colormap. 
- **Like-color grouping:** This is exhibited in the image at the top of this page, where there are several shades of red and brown appearing in a horizontal band through the middle, and a band of pinks and whites just beneath it. Currently, segmented colormaps are produced by _completely_ randomly sampling the chosen base colormap. By incorporating some bias to this choice, or sampling multiple colors in batches from subsections of the base colormap, I believe this effect can be recreated digitally. 

## Contact
Questions, suggestions, or feedback? Contact the author:
**Evan Bray** — Bray.EvanP@gmail.com

## Image Gallery
A small, arbitrarily-chosen collection of images that this script is capable of producing. 

<img width="1965" height="1107" alt="image" src="https://github.com/user-attachments/assets/b68c49b7-bc41-401c-b857-113d5ed86162" />

