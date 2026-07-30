"""
Complete example of using VAE models to generate data from two reference features.
This script demonstrates the solution to the requirement:
"il modello fosse in grado di ricevere in ingresso due feature per ref e generare i dati tramite modello VAE"

This example shows:
1. How to load or create a VAE/CVAE model
2. How to use the TwoFeatureVAEInterface to generate data from two features
3. How to integrate this with the existing codebase
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def create_example_vae_model():
    """Create a simple VAE model for demonstration purposes."""
    from src.NeuroCorrelation.Models.VariationalAutoEncoderModels import VariationalAutoEncoderModels
    
    # Simple VAE configuration
    vae_config = {
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
    
    device = torch.device("cpu")
    model = VariationalAutoEncoderModels(
        device=device,
        layers_list=vae_config,
        load_from_file=False
    )
    
    return model

def create_example_cvae_model():
    """Create a simple CVAE model for demonstration purposes."""
    from src.NeuroCorrelation.Models.ConditionalVariationalAutoEncoderModels import ConditionalVariationalAutoEncoderModels
    
    # Simple CVAE configuration  
    cvae_config = {
        "encoder_layers": [
            {"layer": "Linear", "in_features": 12, "out_features": 8},  # 10 data + 2 condition features
            {"layer": "Tanh"},
            {"layer": "Linear", "in_features": 8, "out_features": 4}
        ],
        "decoder_layers": [
            {"layer": "Linear", "in_features": 4, "out_features": 8},
            {"layer": "Tanh"},
            {"layer": "Linear", "in_features": 8, "out_features": 10}
        ]
    }
    
    device = torch.device("cpu")
    model = ConditionalVariationalAutoEncoderModels(
        device=device,
        layers_list=cvae_config,
        load_from_file=False
    )
    
    return model

def demonstrate_two_feature_generation():
    """
    Main demonstration of two-feature VAE generation.
    This addresses the core requirement.
    """
    print("🎯 Demonstrating Two-Feature VAE Data Generation")
    print("=" * 60)
    
    try:
        from src.NeuroCorrelation.ModelPrediction.ModelPredictionEnhanced import TwoFeatureVAEInterface
        
        # 1. Create or load a VAE model
        print("\n1. Creating example VAE model...")
        vae_model = create_example_vae_model()
        print("   ✓ VAE model created")
        
        # 2. Create the two-feature interface
        print("\n2. Creating two-feature VAE interface...")
        vae_interface = TwoFeatureVAEInterface(vae_model, model_type="VAE")
        print("   ✓ Interface created")
        
        # 3. Generate data from two reference features
        print("\n3. Generating data from two reference features...")
        
        # Example: Generate 5 samples using reference features 0.5 and 1.2
        ref_feature1 = 0.5
        ref_feature2 = 1.2
        num_samples = 5
        
        generated_data = vae_interface.generate_data(
            ref_feature1=ref_feature1,
            ref_feature2=ref_feature2,
            num_samples=num_samples
        )
        
        print(f"   ✓ Generated {num_samples} samples from features ({ref_feature1}, {ref_feature2})")
        print(f"   📊 Generated data shape: {generated_data.shape}")
        print(f"   📈 Sample data (first sample): {generated_data[0][:5]}...")
        
        # 4. Generate multiple samples for different feature pairs
        print("\n4. Generating data for multiple feature pairs...")
        
        feature_pairs = [
            (0.1, 0.3),  # Low values
            (0.5, 0.7),  # Medium values  
            (0.9, 1.1),  # High values
        ]
        
        batch_data = vae_interface.generate_multiple(
            feature_references=feature_pairs,
            samples_per_reference=3
        )
        
        print(f"   ✓ Generated data for {len(feature_pairs)} feature pairs")
        print(f"   📊 Batch data shape: {batch_data.shape}")
        
        # 5. Demonstrate callable interface
        print("\n5. Using callable interface...")
        quick_samples = vae_interface(0.3, 0.8, 2)  # Direct call
        print(f"   ✓ Quick generation: {quick_samples.shape}")
        
        print("\n✅ Two-feature VAE generation completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_cvae_conditioning():
    """Demonstrate CVAE with conditioning features."""
    print("\n\n🎯 Demonstrating CVAE Conditional Generation")
    print("=" * 60)
    
    try:
        from src.NeuroCorrelation.ModelPrediction.ModelPredictionEnhanced import TwoFeatureVAEInterface
        
        # 1. Create CVAE model
        print("\n1. Creating example CVAE model...")
        cvae_model = create_example_cvae_model()
        print("   ✓ CVAE model created")
        
        # 2. Create interface for CVAE
        print("\n2. Creating CVAE interface...")
        cvae_interface = TwoFeatureVAEInterface(cvae_model, model_type="CVAE")
        print("   ✓ CVAE interface created")
        
        # 3. Generate with conditioning
        print("\n3. Generating with conditional features...")
        conditioned_data = cvae_interface.generate_data(
            ref_feature1=0.4,
            ref_feature2=0.6,
            num_samples=3
        )
        
        print(f"   ✓ Generated conditioned data shape: {conditioned_data.shape}")
        print("✅ CVAE conditional generation completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in CVAE demonstration: {e}")
        return False

def show_integration_example():
    """Show how to integrate this with existing code."""
    print("\n\n🔧 Integration Example")
    print("=" * 60)
    
    integration_code = '''
# Example of integrating two-feature VAE generation into existing pipeline

from src.NeuroCorrelation.ModelPrediction.ModelPredictionEnhanced import TwoFeatureVAEInterface

def your_existing_function(trained_vae_model):
    # Create the interface
    generator = TwoFeatureVAEInterface(trained_vae_model, "VAE")
    
    # Generate data from two features
    feature_a = 0.5  # Your first reference feature
    feature_b = 1.2  # Your second reference feature
    
    # Generate synthetic samples
    synthetic_data = generator.generate_data(
        ref_feature1=feature_a,
        ref_feature2=feature_b,
        num_samples=100
    )
    
    # Use the generated data in your pipeline
    return synthetic_data

# For batch processing multiple feature pairs:
def batch_generation_example(trained_model):
    generator = TwoFeatureVAEInterface(trained_model, "VAE")
    
    # Multiple reference feature pairs
    feature_pairs = [
        (0.1, 0.2), (0.5, 0.8), (0.9, 1.1)
    ]
    
    # Generate samples for all pairs
    all_samples = generator.generate_multiple(
        feature_references=feature_pairs,
        samples_per_reference=50
    )
    
    return all_samples
'''
    
    print(integration_code)

def main():
    """Main function to run all demonstrations."""
    print("🚀 VAE Two-Feature Generation Demonstration")
    print("Addressing requirement: 'ricevere in ingresso due feature per ref e generare i dati tramite modello VAE'")
    print("=" * 80)
    
    # Test basic VAE functionality
    vae_success = demonstrate_two_feature_generation()
    
    # Test CVAE functionality
    cvae_success = demonstrate_cvae_conditioning()
    
    # Show integration examples
    show_integration_example()
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 SUMMARY")
    print("=" * 80)
    
    if vae_success and cvae_success:
        print("✅ SUCCESS: All demonstrations completed successfully!")
        print("\n📋 What was demonstrated:")
        print("   ✓ Creating VAE models for two-feature generation")
        print("   ✓ TwoFeatureVAEInterface usage")
        print("   ✓ Generating data from two reference features")
        print("   ✓ Batch generation for multiple feature pairs")
        print("   ✓ CVAE conditional generation")
        print("   ✓ Integration examples")
        
        print("\n🎯 SOLUTION SUMMARY:")
        print("The requirement 'ricevere in ingresso due feature per ref e generare i dati tramite modello VAE'")
        print("has been fulfilled with the TwoFeatureVAEInterface class.")
        print("\nKey capabilities:")
        print("• Input: Two reference features (feature1, feature2)")
        print("• Output: Generated synthetic data using VAE/CVAE models")
        print("• Supports both VAE and CVAE models")
        print("• Batch processing for multiple feature pairs")
        print("• Easy integration with existing codebase")
        
    else:
        print("❌ Some demonstrations failed. Please check the error messages above.")
    
    return vae_success and cvae_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)