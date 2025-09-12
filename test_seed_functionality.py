#!/usr/bin/env python3
"""
Test script to demonstrate and validate seed functionality in paint_pour_tools.
This test ensures that two images with the same seed produce identical results,
even when some parameters are specified manually vs randomly generated.
"""

import numpy as np
import paint_pour_tools as pptools
import matplotlib.pyplot as plt


def test_seed_determinism():
    """
    Test that demonstrates the seed issue and validates the fix.
    Creates two PaintPour objects with the same seed but different parameter specifications.
    They should produce identical final results.
    """
    print("Testing seed determinism...")
    
    # Test parameters
    test_seed = 12345
    image_dimensions = [200, 200]  # Small for faster testing
    
    # Create first image with minimal parameter specification
    print("\nCreating first image with default parameters...")
    paint_pour1 = pptools.PaintPour(
        image_dimensions=image_dimensions,
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False
    )
    image1 = paint_pour1.generate()
    
    # Create second image with some manually specified parameters
    print("\nCreating second image with some manually specified parameters...")
    paint_pour2 = pptools.PaintPour(
        image_dimensions=image_dimensions,
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False,
        # Manually specify some parameters that would normally be random
        num_colormap_levels=30,  # This should normally be randomly chosen
        cmap_name='viridis'      # This should normally be randomly chosen
    )
    image2 = paint_pour2.generate()
    
    # Compare the images
    print("\nComparing results...")
    print(f"Image 1 shape: {image1.shape}")
    print(f"Image 2 shape: {image2.shape}")
    print(f"Images identical: {np.array_equal(image1, image2)}")
    print(f"Max difference: {np.max(np.abs(image1 - image2))}")
    
    # Compare key parameters that should be identical
    print(f"\nParameter comparison:")
    print(f"Seed 1: {paint_pour1.seed}, Seed 2: {paint_pour2.seed}")
    print(f"Octave powers 1: {paint_pour1.octave_powers}")
    print(f"Octave powers 2: {paint_pour2.octave_powers}")
    print(f"Stretch value 1: {paint_pour1.stretch_value}")
    print(f"Stretch value 2: {paint_pour2.stretch_value}")
    print(f"Rescaling exponent 1: {paint_pour1.rescaling_exponent}")
    print(f"Rescaling exponent 2: {paint_pour2.rescaling_exponent}")
    
    # Test should pass when seed functionality is working correctly
    success = np.allclose(image1, image2, rtol=1e-10, atol=1e-10)
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    
    return success, image1, image2, paint_pour1, paint_pour2


def test_colormap_seed_determinism():
    """
    Test colormap generation specifically, as this is done later in the process
    and involves random color/node selection.
    """
    print("\n" + "="*50)
    print("Testing colormap seed determinism...")
    
    test_seed = 54321
    
    # Create two identical paint pour objects
    paint_pour1 = pptools.PaintPour(
        image_dimensions=[100, 100],
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False,
        num_colormap_levels=40
    )
    
    paint_pour2 = pptools.PaintPour(
        image_dimensions=[100, 100], 
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False,
        num_colormap_levels=40
    )
    
    # Generate the colormaps (this happens in generate())
    paint_pour1.pick_paint_pour_colormap()
    paint_pour2.pick_paint_pour_colormap()
    
    # Check if colormaps are identical
    cmap1_colors = paint_pour1.final_colormap(np.linspace(0, 1, 100))
    cmap2_colors = paint_pour2.final_colormap(np.linspace(0, 1, 100))
    
    colormaps_identical = np.allclose(cmap1_colors, cmap2_colors)
    print(f"Colormaps identical: {colormaps_identical}")
    
    return colormaps_identical


if __name__ == "__main__":
    print("Paint Pour Seed Functionality Test")
    print("="*50)
    
    # Run the main test
    success, img1, img2, pp1, pp2 = test_seed_determinism()
    
    # Run colormap specific test
    cmap_success = test_colormap_seed_determinism()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Main seed test: {'PASSED' if success else 'FAILED'}")
    print(f"Colormap seed test: {'PASSED' if cmap_success else 'FAILED'}")
    print(f"Overall: {'PASSED' if (success and cmap_success) else 'FAILED'}")
    
    if not success:
        print("\nThe seed functionality needs to be fixed!")
        print("Random number generation sequence is not deterministic.")
    else:
        print("\nSeed functionality is working correctly!")