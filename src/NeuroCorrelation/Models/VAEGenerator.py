"""
VAEGenerator: A module for generating synthetic data using trained VAE/CVAE models
with two-feature reference input capability.

This module allows users to:
1. Load a trained VAE or CVAE model
2. Provide two reference features as input
3. Generate synthetic data samples based on these features

Usage:
    # For unconditional VAE
    generator = VAEGenerator(model_path="path/to/vae_model.pth", model_type="VAE")
    samples = generator.generate_from_features(feature1=0.5, feature2=1.2, num_samples=10)
    
    # For conditional VAE  
    generator = VAEGenerator(model_path="path/to/cvae_model.pth", model_type="CVAE")
    samples = generator.generate_from_features(feature1=0.5, feature2=1.2, num_samples=10)
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional

class VAEGenerator:
    """
    A generator class for creating synthetic data using trained VAE/CVAE models
    with support for two-feature reference input.
    """
    
    def __init__(self, model=None, model_path: Optional[str] = None, model_type: str = "VAE", 
                 device: str = "cpu", config_path: Optional[str] = None):
        """
        Initialize the VAE Generator.
        
        Args:
            model: Pre-loaded model instance (VAE or CVAE)
            model_path: Path to saved model state dict
            model_type: Type of model ("VAE" or "CVAE")
            device: Device to run the model on ("cpu" or "cuda")
            config_path: Path to model configuration JSON file
        """
        self.model = model
        self.model_path = model_path
        self.model_type = model_type
        self.device = torch.device(device)
        self.config_path = config_path
        
        if model is not None:
            self.model.to(self.device)
            self.model.eval()
        elif model_path is not None:
            self._load_model()
    
    def _load_model(self):
        """Load a model from the provided path."""
        if self.model_path and Path(self.model_path).exists():
            # Load model state dict
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # You would need to instantiate the correct model architecture here
            # This is a placeholder - in practice, you'd need the model configuration
            # to recreate the model architecture
            raise NotImplementedError(
                "Model loading from file requires model architecture configuration. "
                "Please provide a pre-instantiated model or implement model loading."
            )
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
    
    def generate_from_features(self, feature1: float, feature2: float, 
                             num_samples: int = 1, 
                             generation_mode: str = "latent_sampling") -> np.ndarray:
        """
        Generate synthetic data samples based on two reference features.
        
        Args:
            feature1: First reference feature value
            feature2: Second reference feature value  
            num_samples: Number of samples to generate
            generation_mode: Generation strategy ("latent_sampling", "direct_conditioning")
            
        Returns:
            Generated samples as numpy array of shape (num_samples, output_features)
        """
        if self.model is None:
            raise ValueError("No model loaded. Please provide a model or model path.")
        
        if generation_mode == "latent_sampling":
            return self._generate_via_latent_sampling(feature1, feature2, num_samples)
        elif generation_mode == "direct_conditioning":
            return self._generate_via_direct_conditioning(feature1, feature2, num_samples)
        else:
            raise ValueError(f"Unknown generation mode: {generation_mode}")
    
    def _generate_via_latent_sampling(self, feature1: float, feature2: float, 
                                    num_samples: int) -> np.ndarray:
        """
        Generate samples by sampling from the latent space and using the decoder.
        This method creates samples by:
        1. Creating a condition vector from the two features
        2. Sampling from the latent space  
        3. Using the decoder to generate output
        """
        with torch.no_grad():
            # Create condition vector from the two features
            condition = torch.tensor([[feature1, feature2]], dtype=torch.float32).to(self.device)
            
            if self.model_type == "CVAE":
                # For CVAE, we can condition the generation
                # Get the latent dimension from the model
                latent_dim = self._get_latent_dimension()
                
                # Sample from standard normal distribution
                z_samples = torch.randn(num_samples, latent_dim).to(self.device)
                
                # Generate samples using the decoder
                decoder = self.model.get_decoder()
                generated_samples = []
                
                for i in range(num_samples):
                    z_sample = z_samples[i:i+1]  # Keep batch dimension
                    # Note: This is a simplified approach. In practice, you might need
                    # to concatenate condition with z or handle it differently
                    # based on your CVAE architecture
                    output = decoder(z_sample)
                    generated_samples.append(output["x_output"]['data'])
                
                # Stack all samples
                result = torch.cat(generated_samples, dim=0)
                
            elif self.model_type == "VAE":
                # For standard VAE, we'll use the two features to create a reference point
                # and sample around it in the latent space
                
                # Convert features to input format expected by the model
                # This is a placeholder - you'd need to adapt based on your input format
                input_features = self._features_to_input(feature1, feature2)
                
                # Encode to get latent representation
                encoded_output = self.model.models['encoder'](input_features)
                mu = self.model.fc_mu(encoded_output["x_output"]['data'])
                logvar = self.model.fc_logvar(encoded_output["x_output"]['data'])
                
                # Sample multiple times from the learned distribution
                generated_samples = []
                for i in range(num_samples):
                    z = self.model.reparameterize(mu, logvar)
                    output = self.model.models['decoder'](z)
                    generated_samples.append(output["x_output"]['data'])
                
                result = torch.cat(generated_samples, dim=0)
            
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            return result.cpu().numpy()
    
    def _generate_via_direct_conditioning(self, feature1: float, feature2: float, 
                                        num_samples: int) -> np.ndarray:
        """
        Generate samples by directly conditioning the model on the two features.
        This method assumes the model can directly use the features as conditioning input.
        """
        with torch.no_grad():
            # Create a dummy input (this might need to be adapted based on your model)
            # The model should be able to generate based on the condition alone
            
            if self.model_type != "CVAE":
                raise ValueError("Direct conditioning is only supported for CVAE models")
            
            # Create condition vector
            condition = torch.tensor([[feature1, feature2]], dtype=torch.float32).to(self.device)
            
            # For CVAE, we need to provide some input data
            # This is a placeholder approach - you might need to modify based on your architecture
            latent_dim = self._get_latent_dimension()
            
            generated_samples = []
            for i in range(num_samples):
                # Sample from the prior in latent space
                z = torch.randn(1, latent_dim).to(self.device)
                
                # Use decoder to generate sample
                decoder = self.model.get_decoder()
                output = decoder(z)
                generated_samples.append(output["x_output"]['data'])
            
            result = torch.cat(generated_samples, dim=0)
            return result.cpu().numpy()
    
    def _features_to_input(self, feature1: float, feature2: float) -> torch.Tensor:
        """
        Convert two features to the input format expected by the model.
        This is a placeholder method that should be adapted based on your specific
        model input requirements.
        """
        # This is a simple concatenation approach
        # You may need to modify this based on your model's expected input format
        input_tensor = torch.tensor([[feature1, feature2]], dtype=torch.float32).to(self.device)
        
        # If your model expects a specific input size, you might need to pad or transform
        # For example, if the model expects 78 features but you only have 2:
        # padded_input = torch.zeros(1, 78).to(self.device)
        # padded_input[0, :2] = input_tensor[0]
        # return padded_input
        
        return input_tensor
    
    def _get_latent_dimension(self) -> int:
        """
        Get the latent dimension of the model.
        This is a helper method to determine the latent space size.
        """
        if hasattr(self.model, 'fc_mu'):
            return self.model.fc_mu.out_features
        else:
            # Fallback to a reasonable default
            return 64
    
    def generate_samples_batch(self, features_list: List[Tuple[float, float]], 
                              num_samples_per_feature: int = 1) -> np.ndarray:
        """
        Generate samples for multiple feature pairs.
        
        Args:
            features_list: List of (feature1, feature2) tuples
            num_samples_per_feature: Number of samples to generate per feature pair
            
        Returns:
            Generated samples array of shape (len(features_list) * num_samples_per_feature, output_features)
        """
        all_samples = []
        
        for feature1, feature2 in features_list:
            samples = self.generate_from_features(
                feature1, feature2, num_samples_per_feature
            )
            all_samples.append(samples)
        
        return np.vstack(all_samples)
    
    def set_model(self, model):
        """Set a new model for the generator."""
        self.model = model
        if model is not None:
            self.model.to(self.device)
            self.model.eval()
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "model_type": self.model_type,
            "device": str(self.device),
            "latent_dimension": self._get_latent_dimension(),
        }
        
        if hasattr(self.model, 'get_size'):
            info["model_sizes"] = self.model.get_size()
        
        return info


def create_vae_generator_from_trained_model(model, model_type: str = "VAE") -> VAEGenerator:
    """
    Convenience function to create a VAEGenerator from a trained model.
    
    Args:
        model: Trained VAE or CVAE model
        model_type: Type of model ("VAE" or "CVAE")
        
    Returns:
        VAEGenerator instance
    """
    return VAEGenerator(model=model, model_type=model_type)


# Example usage function
def example_usage():
    """
    Example of how to use the VAEGenerator class.
    This is for demonstration purposes.
    """
    print("VAEGenerator Example Usage:")
    print("1. Create generator with pre-trained model:")
    print("   generator = VAEGenerator(model=my_trained_vae, model_type='VAE')")
    print("")
    print("2. Generate samples from two features:")
    print("   samples = generator.generate_from_features(feature1=0.5, feature2=1.2, num_samples=10)")
    print("")
    print("3. Batch generation:")
    print("   features = [(0.1, 0.2), (0.5, 0.8), (0.9, 1.1)]")
    print("   samples = generator.generate_samples_batch(features, num_samples_per_feature=5)")
    print("")
    print("Note: Adapt the _features_to_input method based on your model's input requirements.")


if __name__ == "__main__":
    example_usage()