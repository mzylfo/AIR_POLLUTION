"""
Test script for VAEGenerator functionality.
This script demonstrates how to use the VAEGenerator to create synthetic data
from two reference features using trained VAE/CVAE models.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    import torch
    import torch.nn as nn
    import numpy as np
    from src.NeuroCorrelation.Models.VAEGenerator import VAEGenerator, create_vae_generator_from_trained_model
    from src.NeuroCorrelation.Models.VariationalAutoEncoderModels import VariationalAutoEncoderModels
    from src.NeuroCorrelation.Models.ConditionalVariationalAutoEncoderModels import ConditionalVariationalAutoEncoderModels
    
    def create_simple_vae_config():
        """Create a simple VAE configuration for testing."""
        return {
            "encoder_layers": [
                {"layer": "Linear", "in_features": 10, "out_features": 8},
                {"layer": "Tanh"},
                {"layer": "Linear", "in_features": 8, "out_features": 4}
            ],
            "decoder_layers": [
                {"layer": "Linear", "in_features": 4, "out_features": 8},
                {"layer": "Tanh"},
                {"layer": "Linear", "in_features": 8, "out_features": 10}
            ]
        }
    
    def create_simple_cvae_config():
        """Create a simple CVAE configuration for testing."""
        return {
            "encoder_layers": [
                {"layer": "Linear", "in_features": 12, "out_features": 8},  # 10 + 2 for condition
                {"layer": "Tanh"},
                {"layer": "Linear", "in_features": 8, "out_features": 4}
            ],
            "decoder_layers": [
                {"layer": "Linear", "in_features": 4, "out_features": 8},
                {"layer": "Tanh"},
                {"layer": "Linear", "in_features": 8, "out_features": 10}
            ]
        }
    
    def test_vae_generator():
        """Test the VAE generator functionality."""
        print("Testing VAE Generator...")
        
        # Test 1: Create a simple VAE model
        print("\n1. Creating a simple VAE model for testing...")
        device = torch.device("cpu")
        vae_config = {"VAE": create_simple_vae_config()}
        
        try:
            vae_model = VariationalAutoEncoderModels(
                device=device,
                layers_list=vae_config["VAE"],
                load_from_file=False
            )
            print("   ✓ VAE model created successfully")
        except Exception as e:
            print(f"   ✗ Failed to create VAE model: {e}")
            return False
        
        # Test 2: Create VAE generator
        print("\n2. Creating VAE generator...")
        try:
            vae_generator = create_vae_generator_from_trained_model(vae_model, "VAE")
            print("   ✓ VAE generator created successfully")
        except Exception as e:
            print(f"   ✗ Failed to create VAE generator: {e}")
            return False
        
        # Test 3: Get model info
        print("\n3. Getting model information...")
        try:
            model_info = vae_generator.get_model_info()
            print(f"   ✓ Model info: {model_info}")
        except Exception as e:
            print(f"   ✗ Failed to get model info: {e}")
            return False
        
        # Test 4: Generate samples from features
        print("\n4. Generating samples from two features...")
        try:
            # Generate samples using two reference features
            samples = vae_generator.generate_from_features(
                feature1=0.5, 
                feature2=1.2, 
                num_samples=3,
                generation_mode="latent_sampling"
            )
            print(f"   ✓ Generated samples shape: {samples.shape}")
            print(f"   Sample values (first sample): {samples[0][:5]}...")  # Show first 5 values
        except Exception as e:
            print(f"   ✗ Failed to generate samples: {e}")
            return False
        
        # Test 5: Batch generation
        print("\n5. Testing batch generation...")
        try:
            features_list = [(0.1, 0.2), (0.5, 0.8), (0.9, 1.1)]
            batch_samples = vae_generator.generate_samples_batch(
                features_list, 
                num_samples_per_feature=2
            )
            print(f"   ✓ Batch generation successful, shape: {batch_samples.shape}")
        except Exception as e:
            print(f"   ✗ Failed batch generation: {e}")
            return False
        
        return True
    
    def test_cvae_generator():
        """Test the CVAE generator functionality."""
        print("\n\nTesting CVAE Generator...")
        
        # Test 1: Create a simple CVAE model
        print("\n1. Creating a simple CVAE model for testing...")
        device = torch.device("cpu")
        cvae_config = {"CVAE": create_simple_cvae_config()}
        
        try:
            cvae_model = ConditionalVariationalAutoEncoderModels(
                device=device,
                layers_list=cvae_config["CVAE"],
                load_from_file=False
            )
            print("   ✓ CVAE model created successfully")
        except Exception as e:
            print(f"   ✗ Failed to create CVAE model: {e}")
            return False
        
        # Test 2: Create CVAE generator
        print("\n2. Creating CVAE generator...")
        try:
            cvae_generator = create_vae_generator_from_trained_model(cvae_model, "CVAE")
            print("   ✓ CVAE generator created successfully")
        except Exception as e:
            print(f"   ✗ Failed to create CVAE generator: {e}")
            return False
        
        # Test 3: Generate samples from features
        print("\n3. Generating samples from two features (direct conditioning)...")
        try:
            samples = cvae_generator.generate_from_features(
                feature1=0.3, 
                feature2=0.7, 
                num_samples=2,
                generation_mode="direct_conditioning"
            )
            print(f"   ✓ Generated samples shape: {samples.shape}")
        except Exception as e:
            print(f"   ✗ Failed to generate samples: {e}")
            return False
        
        return True
    
    def main():
        """Main test function."""
        print("=" * 60)
        print("VAE Generator Test Suite")
        print("=" * 60)
        
        # Test basic VAE functionality
        vae_success = test_vae_generator()
        
        # Test CVAE functionality
        cvae_success = test_cvae_generator()
        
        # Summary
        print("\n" + "=" * 60)
        print("Test Results Summary:")
        print(f"VAE Generator: {'✓ PASSED' if vae_success else '✗ FAILED'}")
        print(f"CVAE Generator: {'✓ PASSED' if cvae_success else '✗ FAILED'}")
        print("=" * 60)
        
        if vae_success and cvae_success:
            print("\n🎉 All tests passed! The VAE generator is ready to use.")
            print("\nUsage Examples:")
            print("1. For generating data from two features with VAE:")
            print("   generator = VAEGenerator(model=your_vae_model, model_type='VAE')")
            print("   samples = generator.generate_from_features(0.5, 1.2, num_samples=10)")
            print("\n2. For generating data from two features with CVAE:")
            print("   generator = VAEGenerator(model=your_cvae_model, model_type='CVAE')")
            print("   samples = generator.generate_from_features(0.5, 1.2, num_samples=10)")
        else:
            print("\n❌ Some tests failed. Please check the error messages above.")
        
        return vae_success and cvae_success
    
    if __name__ == "__main__":
        success = main()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all required dependencies are installed.")
    print("You can install them using: pip install torch numpy")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)