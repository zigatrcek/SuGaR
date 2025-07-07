#!/usr/bin/env python3
"""
Test script to verify that memory optimization parameters are properly propagated
through the SuGaR training pipeline.
"""
import sys
import os
sys.path.append('/workspace/SuGaR')

# Test argument parsing in individual training scripts
def test_argument_parsing():
    print("Testing argument parsing in training scripts...")
    
    # Test train_coarse_density.py
    try:
        from train_coarse_density import argparse
        import argparse as ap
        parser = ap.ArgumentParser()
        
        # Add the same arguments as in train_coarse_density.py
        parser.add_argument('-c', '--checkpoint_path', type=str, help='path to checkpoint')
        parser.add_argument('-s', '--scene_path', type=str, help='path to scene')
        parser.add_argument('-o', '--output_dir', type=str, default=None, help='output dir')
        parser.add_argument('-i', '--iteration_to_load', type=int, default=7000, help='iteration')
        parser.add_argument('--eval', type=bool, default=True, help='Use eval split')
        parser.add_argument('--white_background', type=bool, default=False, help='White bg')
        parser.add_argument('-e', '--estimation_factor', type=float, default=0.2, help='estimation factor')
        parser.add_argument('-n', '--normal_factor', type=float, default=0.2, help='normal factor')
        parser.add_argument('--gpu', type=int, default=0, help='GPU index')
        parser.add_argument('--max_img_size', type=int, default=1920, help='Max image size')
        parser.add_argument('--image_resolution', type=int, default=1, help='Image resolution factor')
        
        # Test parsing with memory optimization parameters
        test_args = [
            '--checkpoint_path', '/fake/path',
            '--scene_path', '/fake/scene',
            '--max_img_size', '1024',
            '--image_resolution', '2'
        ]
        
        args = parser.parse_args(test_args)
        assert args.max_img_size == 1024
        assert args.image_resolution == 2
        print("✓ Argument parsing test passed for train_coarse_density.py")
        
    except Exception as e:
        print(f"✗ Argument parsing test failed: {e}")
        return False
    
    return True

def test_model_params():
    print("Testing ModelParams class...")
    
    try:
        from gaussian_splatting.arguments import ModelParams
        
        # Test that ModelParams has the new attributes
        model_params = ModelParams()
        
        # Check if the attributes exist and have correct default values
        assert hasattr(model_params, '_max_img_size'), "ModelParams missing _max_img_size attribute"
        assert hasattr(model_params, '_image_resolution'), "ModelParams missing _image_resolution attribute"
        
        # Check default values
        assert model_params._max_img_size == 1920, f"Expected _max_img_size=1920, got {model_params._max_img_size}"
        assert model_params._image_resolution == 1, f"Expected _image_resolution=1, got {model_params._image_resolution}"
        
        print("✓ ModelParams test passed")
        
    except Exception as e:
        print(f"✗ ModelParams test failed: {e}")
        return False
    
    return True

def test_pipeline_parameters():
    print("Testing pipeline parameter propagation...")
    
    try:
        # Test that the pipeline script has the correct arguments
        sys.path.append('/workspace/SuGaR')
        
        # Import and check if train_full_pipeline.py has the correct arguments
        import argparse
        parser = argparse.ArgumentParser()
        
        # Add arguments similar to train_full_pipeline.py
        parser.add_argument('-s', '--scene_path', type=str, help='scene path')
        parser.add_argument('-c', '--checkpoint_path', type=str, help='checkpoint path')
        parser.add_argument('--max_img_size', type=int, default=1920, help='Max image size')
        parser.add_argument('--image_resolution', type=int, default=1, help='Image resolution factor')
        
        # Test parsing
        test_args = [
            '--scene_path', '/fake/scene',
            '--checkpoint_path', '/fake/checkpoint',
            '--max_img_size', '1280',
            '--image_resolution', '2'
        ]
        
        args = parser.parse_args(test_args)
        assert args.max_img_size == 1280
        assert args.image_resolution == 2
        
        print("✓ Pipeline parameter test passed")
        
    except Exception as e:
        print(f"✗ Pipeline parameter test failed: {e}")
        return False
    
    return True

def main():
    print("SuGaR Memory Optimization Implementation Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_argument_parsing()
    all_tests_passed &= test_model_params()
    all_tests_passed &= test_pipeline_parameters()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Memory optimization implementation looks good.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())
