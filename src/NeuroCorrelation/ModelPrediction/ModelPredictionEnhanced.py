"""
Enhanced ModelPrediction class with two-feature generation capability.
This extends the existing ModelPrediction functionality to support
direct generation from two reference features.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path

# Import the new VAEGenerator
from .VAEGenerator import VAEGenerator


class ModelPredictionEnhanced:
    """
    Enhanced prediction class that extends the original ModelPrediction
    with two-feature reference generation capability.
    """
    
    def __init__(self, model, device, model_type="VAE", **kwargs):
        """
        Initialize the enhanced prediction class.
        
        Args:
            model: Trained VAE/CVAE model
            device: Device to run on
            model_type: Type of model ("VAE" or "CVAE")
            **kwargs: Additional arguments for compatibility
        """
        self.model = model
        self.device = device
        self.model_type = model_type
        
        # Initialize the VAE generator
        self.vae_generator = VAEGenerator(
            model=model,
            model_type=model_type,
            device=device
        )
    
    def generate_from_two_features(self, feature1: float, feature2: float, 
                                 num_samples: int = 1) -> np.ndarray:
        """
        Generate synthetic data samples based on two reference features.
        This is the main method to fulfill the requirement of receiving
        two features as input and generating data via the VAE model.
        
        Args:
            feature1: First reference feature value
            feature2: Second reference feature value
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples as numpy array
        """
        return self.vae_generator.generate_from_features(
            feature1=feature1,
            feature2=feature2,
            num_samples=num_samples,
            generation_mode="latent_sampling"
        )
    
    def generate_samples_for_feature_pairs(self, feature_pairs: List[Tuple[float, float]], 
                                         samples_per_pair: int = 1) -> np.ndarray:
        """
        Generate samples for multiple feature pairs.
        
        Args:
            feature_pairs: List of (feature1, feature2) tuples
            samples_per_pair: Number of samples per feature pair
            
        Returns:
            Generated samples array
        """
        return self.vae_generator.generate_samples_batch(
            features_list=feature_pairs,
            num_samples_per_feature=samples_per_pair
        )
    
    def generate_with_conditioning(self, feature1: float, feature2: float, 
                                 num_samples: int = 1) -> np.ndarray:
        """
        Generate samples using direct conditioning (for CVAE models).
        
        Args:
            feature1: First conditioning feature
            feature2: Second conditioning feature
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples array
        """
        if self.model_type != "CVAE":
            raise ValueError("Direct conditioning is only available for CVAE models")
        
        return self.vae_generator.generate_from_features(
            feature1=feature1,
            feature2=feature2,
            num_samples=num_samples,
            generation_mode="direct_conditioning"
        )
    
    def get_generation_info(self) -> dict:
        """Get information about the generation capabilities."""
        return self.vae_generator.get_model_info()


class TwoFeatureVAEInterface:
    """
    A simplified interface specifically for the two-feature VAE generation requirement.
    This class provides a clean, focused interface for the specific use case.
    """
    
    def __init__(self, vae_model, model_type: str = "VAE", device: str = "cpu"):
        """
        Initialize the two-feature VAE interface.
        
        Args:
            vae_model: Trained VAE or CVAE model
            model_type: Type of model ("VAE" or "CVAE")
            device: Device to run on
        """
        self.model = vae_model
        self.model_type = model_type
        self.device = torch.device(device)
        self.generator = VAEGenerator(model=vae_model, model_type=model_type, device=device)
    
    def generate_data(self, ref_feature1: float, ref_feature2: float, 
                     num_samples: int = 1) -> np.ndarray:
        """
        Main method to generate data from two reference features.
        This directly addresses the requirement: "ricevere in ingresso due feature per ref 
        e generare i dati tramite modello VAE"
        
        Args:
            ref_feature1: First reference feature value
            ref_feature2: Second reference feature value
            num_samples: Number of data samples to generate
            
        Returns:
            Generated data samples as numpy array
        """
        return self.generator.generate_from_features(
            feature1=ref_feature1,
            feature2=ref_feature2,
            num_samples=num_samples
        )
    
    def generate_multiple(self, feature_references: List[Tuple[float, float]], 
                         samples_per_reference: int = 1) -> np.ndarray:
        """
        Generate data for multiple reference feature pairs.
        
        Args:
            feature_references: List of (feature1, feature2) reference pairs
            samples_per_reference: Number of samples to generate per reference pair
            
        Returns:
            All generated samples stacked in a numpy array
        """
        return self.generator.generate_samples_batch(
            features_list=feature_references,
            num_samples_per_feature=samples_per_reference
        )
    
    def set_generation_mode(self, mode: str):
        """
        Set the generation mode for subsequent calls.
        
        Args:
            mode: "latent_sampling" or "direct_conditioning"
        """
        self.generation_mode = mode
    
    def __call__(self, feature1: float, feature2: float, num_samples: int = 1) -> np.ndarray:
        """
        Make the interface callable for convenient usage.
        
        Example:
            vae_interface = TwoFeatureVAEInterface(model)
            samples = vae_interface(0.5, 1.2, num_samples=10)
        """
        return self.generate_data(feature1, feature2, num_samples)


def create_two_feature_interface(model, model_type: str = "VAE") -> TwoFeatureVAEInterface:
    """
    Factory function to create a two-feature VAE interface.
    
    Args:
        model: Trained VAE/CVAE model
        model_type: Type of model
        
    Returns:
        TwoFeatureVAEInterface instance
    """
    return TwoFeatureVAEInterface(model, model_type)


# Example usage demonstrations
def demo_usage():
    """
    Demonstrate how to use the two-feature VAE interface.
    """
    print("Two-Feature VAE Interface Usage Examples:")
    print("=" * 50)
    
    print("\n1. Basic usage:")
    print("   interface = TwoFeatureVAEInterface(trained_vae_model)")
    print("   generated_data = interface.generate_data(0.5, 1.2, num_samples=10)")
    
    print("\n2. Callable interface:")
    print("   interface = TwoFeatureVAEInterface(trained_vae_model)")
    print("   generated_data = interface(0.5, 1.2, 10)  # Same as above")
    
    print("\n3. Multiple reference pairs:")
    print("   references = [(0.1, 0.2), (0.5, 0.8), (0.9, 1.1)]")
    print("   all_data = interface.generate_multiple(references, samples_per_reference=5)")
    
    print("\n4. With CVAE conditioning:")
    print("   cvae_interface = TwoFeatureVAEInterface(trained_cvae_model, 'CVAE')")
    print("   cvae_interface.set_generation_mode('direct_conditioning')")
    print("   conditioned_data = cvae_interface(0.3, 0.7, 5)")
    
    print("\n5. Integration with existing code:")
    print("   # In your existing pipeline:")
    print("   vae_generator = create_two_feature_interface(your_trained_model)")
    print("   synthetic_samples = vae_generator(feature_a, feature_b, 100)")


if __name__ == "__main__":
    demo_usage()