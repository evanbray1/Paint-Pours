#!/usr/bin/env python3
"""
Final validation test for seed functionality.
This test confirms that the seed issue has been resolved.
"""

import numpy as np
import paint_pour_tools as pptools


def test_identical_scenarios_with_seed():
    """
    Test that identical scenarios with the same seed produce identical results.
    """
    print("Testing identical scenarios with same seed...")
    
    test_seed = 77777
    image_dimensions = [120, 120]
    
    # Two identical scenarios
    paint_pour1 = pptools.PaintPour(
        image_dimensions=image_dimensions,
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False
    )
    image1 = paint_pour1.generate()
    
    paint_pour2 = pptools.PaintPour(
        image_dimensions=image_dimensions,
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False
    )
    image2 = paint_pour2.generate()
    
    identical = np.array_equal(image1, image2)
    print(f"Identical scenarios produce identical results: {identical}")
    return identical


def test_deterministic_random_sequence():
    """
    Test that the random sequence is deterministic regardless of manual parameters.
    We'll test this by checking that automatically generated parameters are consistent.
    """
    print("\nTesting deterministic random sequence...")
    
    test_seed = 55555
    
    # Scenario A: All automatic
    paint_pour_a = pptools.PaintPour(
        image_dimensions=[100, 100],
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False
    )
    
    # Scenario B: Some manual parameters (but others should be same as A)
    paint_pour_b = pptools.PaintPour(
        image_dimensions=[100, 100],
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False,
        cmap_name='viridis',  # Manual
        num_colormap_levels=30  # Manual
    )
    
    # The automatically generated parameters should be identical
    octaves_match = np.allclose(paint_pour_a.octave_powers, paint_pour_b.octave_powers)
    stretch_match = paint_pour_a.stretch_value == paint_pour_b.stretch_value
    rescale_match = abs(paint_pour_a.rescaling_exponent - paint_pour_b.rescaling_exponent) < 1e-10
    
    print(f"Octave powers match: {octaves_match}")
    print(f"Stretch values match: {stretch_match}")
    print(f"Rescaling exponents match: {rescale_match}")
    print(f"Auto colormap A: {paint_pour_a.cmap_name}, levels A: {paint_pour_a.num_colormap_levels}")
    print(f"Manual colormap B: {paint_pour_b.cmap_name}, levels B: {paint_pour_b.num_colormap_levels}")
    
    return octaves_match and stretch_match and rescale_match


def test_noise_generation_consistency():
    """
    Test that the underlying noise generation is consistent with same parameters and seed.
    """
    print("\nTesting noise generation consistency...")
    
    test_seed = 33333
    
    # Same seed, same octave parameters -> should produce identical noise
    paint_pour1 = pptools.PaintPour(
        image_dimensions=[80, 80],
        seed=test_seed,
        octave_powers=[1, 0.3, 0.05, 0.0],
        stretch_value=0,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False
    )
    _ = paint_pour1.generate()
    
    paint_pour2 = pptools.PaintPour(
        image_dimensions=[80, 80],
        seed=test_seed,
        octave_powers=[1, 0.3, 0.05, 0.0],
        stretch_value=0,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False,
        cmap_name='plasma'  # Different colormap shouldn't affect noise
    )
    _ = paint_pour2.generate()
    
    noise_identical = np.allclose(paint_pour1.noise_map, paint_pour2.noise_map, rtol=1e-12)
    print(f"Noise maps with same parameters are identical: {noise_identical}")
    
    return noise_identical


def test_colormap_generation_determinism():
    """
    Test that colormap generation is deterministic.
    """
    print("\nTesting colormap generation determinism...")
    
    test_seed = 11111
    
    # Multiple paint pours with same seed and colormap settings
    colormap_samples = []
    for i in range(3):
        paint_pour = pptools.PaintPour(
            image_dimensions=[60, 60],
            seed=test_seed,
            num_colormap_levels=25,
            cmap_name='plasma',
            display_final_image=False,
            save_image=False,
            show_intermediate_plots=False
        )
        paint_pour.pick_paint_pour_colormap()
        
        # Sample the colormap
        test_points = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        samples = paint_pour.final_colormap(test_points)
        colormap_samples.append(samples)
    
    # All samples should be identical
    all_identical = all(np.allclose(colormap_samples[0], sample, rtol=1e-12) for sample in colormap_samples[1:])
    print(f"Colormap generation is deterministic: {all_identical}")
    
    return all_identical


def test_original_issue_scenario():
    """
    Test the exact scenario described in the original issue:
    Two images with same seed, one with specified colormap levels, one without.
    """
    print("\nTesting original issue scenario...")
    
    test_seed = 999999
    image_dimensions = [100, 100]
    
    # Image 1: User likes this image (all defaults)
    paint_pour1 = pptools.PaintPour(
        image_dimensions=image_dimensions,
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False
    )
    image1 = paint_pour1.generate()
    
    # Image 2: User wants same image but with different colormap levels
    paint_pour2 = pptools.PaintPour(
        image_dimensions=image_dimensions,
        seed=test_seed,
        num_colormap_levels=50,  # User manually specifies this
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False
    )
    image2 = paint_pour2.generate()
    
    # The underlying structure should be the same, but colormaps different
    # So we compare the noise maps (before colormap application)
    noise_identical = np.allclose(paint_pour1.noise_map, paint_pour2.noise_map)
    surface_identical = np.allclose(paint_pour1.paint_pour_surface, paint_pour2.paint_pour_surface)
    
    # The automatically chosen parameters should be the same
    auto_params_same = (
        np.allclose(paint_pour1.octave_powers, paint_pour2.octave_powers) and
        paint_pour1.stretch_value == paint_pour2.stretch_value and
        abs(paint_pour1.rescaling_exponent - paint_pour2.rescaling_exponent) < 1e-10 and
        paint_pour1.cmap_name == paint_pour2.cmap_name  # Both should get same auto colormap
    )
    
    print(f"Underlying noise identical: {noise_identical}")
    print(f"Paint pour surface identical: {surface_identical}")
    print(f"Auto parameters consistent: {auto_params_same}")
    print(f"Colormap 1: {paint_pour1.cmap_name}, levels: {paint_pour1.num_colormap_levels}")
    print(f"Colormap 2: {paint_pour2.cmap_name}, levels: {paint_pour2.num_colormap_levels}")
    
    return noise_identical and surface_identical and auto_params_same


if __name__ == "__main__":
    print("Final Seed Functionality Validation")
    print("="*50)
    
    test1_pass = test_identical_scenarios_with_seed()
    test2_pass = test_deterministic_random_sequence() 
    test3_pass = test_noise_generation_consistency()
    test4_pass = test_colormap_generation_determinism()
    test5_pass = test_original_issue_scenario()
    
    print("\n" + "="*50)
    print("TEST RESULTS:")
    print(f"1. Identical scenarios: {'PASS' if test1_pass else 'FAIL'}")
    print(f"2. Deterministic sequence: {'PASS' if test2_pass else 'FAIL'}")  
    print(f"3. Noise consistency: {'PASS' if test3_pass else 'FAIL'}")
    print(f"4. Colormap determinism: {'PASS' if test4_pass else 'FAIL'}")
    print(f"5. Original issue scenario: {'PASS' if test5_pass else 'FAIL'}")
    
    all_pass = all([test1_pass, test2_pass, test3_pass, test4_pass, test5_pass])
    print(f"\nOVERALL: {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
    
    if all_pass:
        print("\n🎉 Seed functionality has been successfully fixed!")
        print("Users can now:")
        print("- Use a seed to reproduce an image they like")
        print("- Manually specify individual parameters while keeping others consistent")
        print("- Get deterministic results regardless of parameter specification order")
    else:
        print("\n⚠️  Some issues remain with the seed functionality.")