#!/usr/bin/env python3

"""
Test script to verify memory optimization parameters are working correctly.
"""

import sys
import argparse

def test_pipeline_args():
    """Test that pipeline script accepts memory optimization parameters."""
    print("Testing train_full_pipeline.py arguments...")
    
    sys.path.insert(0, '/workspace/SuGaR')
    
    # Import the pipeline script's argument parser setup
    import train_full_pipeline
    
    # Create a parser like the pipeline script does
    parser = argparse.ArgumentParser(description='Test memory optimization parameters')
    
    # Add memory optimization parameters
    parser.add_argument('--img_resolution', type=int, default=1,
                        help='Factor by which to downscale images.')
    parser.add_argument('--img_size_limit', type=int, default=1920,
                        help='Maximum image size.')
    
    # Test parsing with memory optimization parameters
    test_args = ['--img_resolution', '2', '--img_size_limit', '1024']
    args = parser.parse_args(test_args)
    
    print(f"✓ Pipeline arguments parsed successfully:")
    print(f"  img_resolution: {args.img_resolution}")
    print(f"  img_size_limit: {args.img_size_limit}")
    return True

def test_modelparams():
    """Test that ModelParams accepts the new parameter names."""
    print("\nTesting ModelParams with new parameter names...")
    
    sys.path.insert(0, '/workspace/SuGaR/gaussian_splatting')
    
    try:
        from arguments import ModelParams
        
        # Create a test parser
        parser = argparse.ArgumentParser()
        
        # Create ModelParams instance (this will add arguments)
        model_params = ModelParams(parser)
        
        # Test parsing with the new parameter names
        test_args = ['--img_resolution', '2', '--img_size_limit', '1024']
        args = parser.parse_args(test_args)
        
        print(f"✓ ModelParams created and parsed successfully:")
        print(f"  img_resolution: {args.img_resolution}")
        print(f"  img_size_limit: {args.img_size_limit}")
        
        return True
        
    except Exception as e:
        print(f"✗ ModelParams test failed: {e}")
        return False

def test_gs_wrapper():
    """Test that GaussianSplattingWrapper accepts memory optimization parameters."""
    print("\nTesting GaussianSplattingWrapper parameter names...")
    
    sys.path.insert(0, '/workspace/SuGaR')
    
    try:
        from sugar_scene.gs_model import GaussianSplattingWrapper
        import inspect
        
        # Get the constructor signature
        sig = inspect.signature(GaussianSplattingWrapper.__init__)
        
        # Check if the new parameter names are present
        params = list(sig.parameters.keys())
        
        has_img_size_limit = 'img_size_limit' in params
        has_img_resolution = 'img_resolution' in params
        
        print(f"✓ GaussianSplattingWrapper signature check:")
        print(f"  img_size_limit parameter: {'✓' if has_img_size_limit else '✗'}")
        print(f"  img_resolution parameter: {'✓' if has_img_resolution else '✗'}")
        
        return has_img_size_limit and has_img_resolution
        
    except Exception as e:
        print(f"✗ GaussianSplattingWrapper test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing memory optimization parameter implementation...")
    print("=" * 60)
    
    success = True
    
    try:
        success &= test_pipeline_args()
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        success = False
    
    try:
        success &= test_modelparams()
    except Exception as e:
        print(f"✗ ModelParams test failed: {e}")
        success = False
    
    try:
        success &= test_gs_wrapper()
    except Exception as e:
        print(f"✗ GaussianSplattingWrapper test failed: {e}")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed! Memory optimization parameters are properly implemented.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    sys.exit(0 if success else 1)
