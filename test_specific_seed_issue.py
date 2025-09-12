#!/usr/bin/env python3
"""
Test to specifically demonstrate the seed issue with random calls
that happen in different orders depending on user-specified parameters.
"""

import numpy as np
import paint_pour_tools as pptools


def test_specific_seed_issue():
    """
    This test demonstrates that the random call sequence is not deterministic
    when different parameters are specified manually vs automatically.
    """
    print("Testing specific seed issue...")
    
    test_seed = 42
    
    # Test 1: Everything automatic
    print("\nTest 1: All parameters automatic")
    np.random.seed(test_seed)
    paint_pour1 = pptools.PaintPour(
        image_dimensions=[100, 100],
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False
    )
    
    print(f"Automatically chosen colormap: {paint_pour1.cmap_name}")
    print(f"Automatically chosen num_colormap_levels: {paint_pour1.num_colormap_levels}")
    print(f"Octave powers: {paint_pour1.octave_powers}")
    
    # Test 2: Some parameters manually specified
    print("\nTest 2: Some parameters manually specified")
    np.random.seed(test_seed)  # Reset to same seed
    paint_pour2 = pptools.PaintPour(
        image_dimensions=[100, 100],
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False,
        cmap_name='viridis',  # Manually specify this
        num_colormap_levels=30  # Manually specify this
    )
    
    print(f"Manually specified colormap: {paint_pour2.cmap_name}")
    print(f"Manually specified num_colormap_levels: {paint_pour2.num_colormap_levels}")
    print(f"Octave powers: {paint_pour2.octave_powers}")
    
    # The octave powers should be the same if the seed is working correctly
    octaves_match = np.allclose(paint_pour1.octave_powers, paint_pour2.octave_powers)
    print(f"\nOctave powers match: {octaves_match}")
    
    # Now test what happens when we generate the images
    print("\nGenerating images...")
    
    # Reset seeds again before generating
    np.random.seed(test_seed)
    paint_pour1_fresh = pptools.PaintPour(
        image_dimensions=[100, 100],
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False
    )
    image1 = paint_pour1_fresh.generate()
    
    np.random.seed(test_seed)
    paint_pour2_fresh = pptools.PaintPour(
        image_dimensions=[100, 100],
        seed=test_seed,
        display_final_image=False,
        save_image=False,
        show_intermediate_plots=False,
        cmap_name='viridis',
        num_colormap_levels=30
    )
    image2 = paint_pour2_fresh.generate()
    
    images_match = np.allclose(image1, image2, rtol=1e-10)
    print(f"Images match: {images_match}")
    print(f"Max difference: {np.max(np.abs(image1 - image2))}")
    
    return octaves_match and images_match


def test_random_sequence_issue():
    """
    Test to show exactly where the random sequence breaks.
    """
    print("\n" + "="*50)
    print("Testing random sequence issue step by step...")
    
    test_seed = 123
    
    # Track the sequence of random numbers generated
    print("\nSequence 1: All automatic parameters")
    np.random.seed(test_seed)
    random_calls_1 = []
    
    # Simulate the calls that happen in assign_unspecified_parameters
    temp_seed = np.random.randint(1, int(1e8))
    random_calls_1.append(f"seed: {temp_seed}")
    
    # octave_powers
    temp_octaves = [1,
                   np.round(np.random.uniform(0.1, 0.5), 1),
                   np.round(np.random.uniform(0.0, 0.1), 2),
                   np.random.choice([0.0, 0.01, 0.02, 0.08], p=[0.55, 0.15, 0.15, 0.15])]
    random_calls_1.append(f"octaves: {temp_octaves}")
    
    # stretch_value  
    temp_stretch = np.random.randint(-2, 3)
    random_calls_1.append(f"stretch: {temp_stretch}")
    
    # rescaling_exponent
    temp_rescale = 10 ** np.random.uniform(0.1, 2.6)
    random_calls_1.append(f"rescale: {temp_rescale}")
    
    # cmap_name (this involves more random calls in pick_random_colormap)
    temp_colormap = pptools.pick_random_colormap(show_plot=False)
    random_calls_1.append(f"colormap: {temp_colormap.name}")
    
    # num_colormap_levels
    temp_levels = np.random.choice([30, 40, 50])
    random_calls_1.append(f"levels: {temp_levels}")
    
    # Now the colormap generation calls
    colors = np.random.randint(low=0, high=10000, size=temp_levels)
    nodes = np.sort(np.random.uniform(low=0, high=1, size=len(colors) - 2))
    random_calls_1.append(f"colors: {colors[:5]}...")  # Show first 5
    random_calls_1.append(f"nodes: {nodes[:5]}...")   # Show first 5
    
    print("Random calls sequence 1:")
    for call in random_calls_1:
        print(f"  {call}")
    
    # Now sequence 2: some manual parameters
    print("\nSequence 2: Some manual parameters")
    np.random.seed(test_seed)
    random_calls_2 = []
    
    # Same seed call
    temp_seed = np.random.randint(1, int(1e8))
    random_calls_2.append(f"seed: {temp_seed}")
    
    # Same octave call
    temp_octaves = [1,
                   np.round(np.random.uniform(0.1, 0.5), 1),
                   np.round(np.random.uniform(0.0, 0.1), 2),
                   np.random.choice([0.0, 0.01, 0.02, 0.08], p=[0.55, 0.15, 0.15, 0.15])]
    random_calls_2.append(f"octaves: {temp_octaves}")
    
    # Same stretch call
    temp_stretch = np.random.randint(-2, 3)
    random_calls_2.append(f"stretch: {temp_stretch}")
    
    # Same rescale call
    temp_rescale = 10 ** np.random.uniform(0.1, 2.6)
    random_calls_2.append(f"rescale: {temp_rescale}")
    
    # SKIP colormap random call (it's manually specified)
    # SKIP num_colormap_levels random call (it's manually specified)
    
    # But colormap generation still happens with specified values
    temp_levels = 30  # manually specified
    colors = np.random.randint(low=0, high=10000, size=temp_levels)
    nodes = np.sort(np.random.uniform(low=0, high=1, size=len(colors) - 2))
    random_calls_2.append(f"colors: {colors[:5]}...")  # Show first 5  
    random_calls_2.append(f"nodes: {nodes[:5]}...")   # Show first 5
    
    print("Random calls sequence 2:")
    for call in random_calls_2:
        print(f"  {call}")
    
    print(f"\nSequences match: {random_calls_1 == random_calls_2}")
    return random_calls_1 == random_calls_2


if __name__ == "__main__":
    print("Specific Seed Issue Test")
    print("="*50)
    
    success1 = test_specific_seed_issue()
    success2 = test_random_sequence_issue()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Basic seed test: {'PASSED' if success1 else 'FAILED'}")
    print(f"Random sequence test: {'PASSED' if success2 else 'FAILED'}")
    
    if not success1 or not success2:
        print("\nThe seed issue is confirmed - random sequences are inconsistent!")
    else:
        print("\nSeed functionality is working correctly!")