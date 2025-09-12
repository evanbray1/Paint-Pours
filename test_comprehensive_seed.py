#!/usr/bin/env python3
"""
Updated test to validate that the seed functionality is now working correctly.
This test validates that two images with the same seed produce identical results
regardless of which parameters are specified manually vs automatically.
"""

import numpy as np
import paint_pour_tools as pptools


def test_comprehensive_seed_functionality():
    """
    Comprehensive test that validates seed functionality across different scenarios.
    """
    print("Testing comprehensive seed functionality...")
    
    test_seed = 99999
    image_dimensions = [150, 150]  # Small for faster testing
    
    # Scenario 1: All parameters automatic
    print("\nScenario 1: All parameters automatic")
    paint_pour1 = pptools.PaintPour(
        image_dimensions=image_dimensions,
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False
    )
    image1 = paint_pour1.generate()
    
    # Scenario 2: Some parameters manually specified
    print("\nScenario 2: Some parameters manually specified")
    paint_pour2 = pptools.PaintPour(
        image_dimensions=image_dimensions,
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False,
        num_colormap_levels=40,  # Manually specify
        cmap_name='plasma'       # Manually specify
    )
    image2 = paint_pour2.generate()
    
    # Scenario 3: Different manual parameters but same seed
    print("\nScenario 3: Different manual parameters but same seed")
    paint_pour3 = pptools.PaintPour(
        image_dimensions=image_dimensions,
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False,
        octave_powers=[1, 0.3, 0.05, 0.0],  # Manually specify
        stretch_value=1,                    # Manually specify
        rescaling_exponent=50.0             # Manually specify
    )
    image3 = paint_pour3.generate()
    
    # Compare underlying noise maps (should be identical for same seed)
    print("\nComparing underlying noise structures...")
    noise_match_1_2 = np.allclose(paint_pour1.noise_map, paint_pour2.noise_map, rtol=1e-10)
    noise_match_1_3 = np.allclose(paint_pour1.noise_map, paint_pour3.noise_map, rtol=1e-10)
    
    print(f"Noise maps 1-2 match: {noise_match_1_2}")
    print(f"Noise maps 1-3 match: {noise_match_1_3}")
    
    # Compare key random parameters that should be consistent
    print("\nComparing automatically generated parameters...")
    print(f"Paint pour 1 - cmap: {paint_pour1.cmap_name}, levels: {paint_pour1.num_colormap_levels}")
    print(f"Paint pour 2 - cmap: {paint_pour2.cmap_name}, levels: {paint_pour2.num_colormap_levels}")
    print(f"Paint pour 3 - cmap: {paint_pour3.cmap_name}, levels: {paint_pour3.num_colormap_levels}")
    
    # For parameters that were automatically generated, they should be consistent
    auto_params_match = (
        paint_pour1.cmap_name == paint_pour3.cmap_name and  # Both should have same auto colormap
        paint_pour1.num_colormap_levels == paint_pour3.num_colormap_levels  # Both should have same auto levels
    )
    
    print(f"Automatically generated parameters consistent: {auto_params_match}")
    
    # Test with cells enabled
    print("\nTesting with cells enabled...")
    paint_pour4 = pptools.PaintPour(
        image_dimensions=[100, 100],  # Even smaller for cell test
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False,
        prominent_cells=True
    )
    image4 = paint_pour4.generate()
    
    paint_pour5 = pptools.PaintPour(
        image_dimensions=[100, 100],
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False,
        prominent_cells=True,
        cmap_name='viridis'  # Manual colormap
    )
    image5 = paint_pour5.generate()
    
    # With cells, the underlying structure should still be the same
    cells_match = np.allclose(image4, image5, rtol=1e-8)  # Slightly looser tolerance for cell processing
    print(f"Images with cells match: {cells_match}")
    
    # Overall success
    overall_success = (
        noise_match_1_2 and 
        noise_match_1_3 and 
        auto_params_match and
        cells_match
    )
    
    print(f"\nOverall test success: {overall_success}")
    return overall_success


def test_colormap_determinism():
    """
    Test that colormap generation is deterministic with the same seed.
    """
    print("\n" + "="*50)
    print("Testing colormap determinism...")
    
    test_seed = 88888
    
    # Create multiple PaintPour objects with same seed
    results = []
    for i in range(3):
        paint_pour = pptools.PaintPour(
            image_dimensions=[80, 80],
            seed=test_seed,
            display_final_image=False,
            save_image=False,
            show_intermediate_plots=False,
            num_colormap_levels=35
        )
        paint_pour.pick_paint_pour_colormap()
        
        # Sample the colormap at fixed points
        test_points = np.linspace(0, 1, 10)
        colormap_samples = paint_pour.final_colormap(test_points)
        results.append(colormap_samples)
    
    # All colormap samples should be identical
    all_match = all(np.allclose(results[0], result, rtol=1e-10) for result in results[1:])
    print(f"All colormap samples identical: {all_match}")
    
    return all_match


if __name__ == "__main__":
    print("Comprehensive Seed Functionality Test")
    print("="*50)
    
    success1 = test_comprehensive_seed_functionality()
    success2 = test_colormap_determinism()
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"Comprehensive test: {'PASSED' if success1 else 'FAILED'}")
    print(f"Colormap determinism test: {'PASSED' if success2 else 'FAILED'}")
    print(f"Overall: {'PASSED' if (success1 and success2) else 'FAILED'}")
    
    if success1 and success2:
        print("\n✅ Seed functionality is working correctly!")
        print("Users can now reliably reproduce images with specific seeds")
        print("while manually overriding individual parameters.")
    else:
        print("\n❌ Seed functionality still has issues that need to be addressed.")