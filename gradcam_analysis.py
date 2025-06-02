import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import os
import logging
from networks import MuZeroNet
from env.hanoi import TowersOfHanoi
from MCTS.mcts import MCTS

class GradCAM:
    """Class for Grad-CAM visualization."""
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture gradients and activations."""
        def forward_hook(module, input, output):
            self.activations = output.clone().detach().requires_grad_(True)

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].clone().detach()

        # Find the target layer and register hooks
        target_layer = self._find_layer_by_name(self.model, self.target_layer_name)
        if target_layer is None:
            raise ValueError(f"Layer {self.target_layer_name} not found in model")

        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def _find_layer_by_name(self, model, layer_name):
        """Recursively find a layer by its name in the model."""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def generate_cam(self, input_tensor, class_idx=None, network_type='policy'):
        """Generate the Grad-CAM heatmap."""
        input_tensor = Variable(input_tensor, requires_grad=True)

        h_state = self.model.represent(input_tensor).clone()

        # Forward pass through the model
        if network_type == 'policy':
            output, _ = self.model.prediction(h_state)
            if class_idx is None:
                class_idx = torch.argmax(output).item()
            score = output[0, class_idx]
        elif network_type == 'value':
            _, output = self.model.prediction(h_state)
            score = output[0] if len(output.shape) > 0 else output
        elif network_type == 'reward':
            raise NotImplementedError("Reward network Grad-CAM not implemented")
        else:
            raise ValueError("network_type must be 'policy', 'value', or 'reward'")
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Generate CAM
        gradients = self.gradients.clone() if self.gradients is not None else None
        activations = self.activations.clone() if self.activations is not None else None

        if gradients is None or activations is None:
            raise RuntimeError("Gradients or activations not captured. Check target layer name.")
        
        logging.info(f"Gradients shape: {gradients.shape}, Activations shape: {activations.shape}")
        logging.info(f"Gradients range: [{gradients.min():.6f}, {gradients.max():.6f}]")
        logging.info(f"Activations range: [{activations.min():.6f}, {activations.max():.6f}]")

        weights = torch.mean(gradients, dim=tuple(range(2, gradients.dim())), keepdim=True) if gradients.dim() > 2 else gradients

        logging.info(f"Weights shape: {weights.shape}")
        logging.info(f"Weights range: [{weights.min():.6f}, {weights.max():.6f}]")

        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam = cam.squeeze().cpu().detach().numpy()

        logging.info(f"Final CAM: shape={cam.shape}, range=[{cam.min():.6f}, {cam.max():.6f}]")

        if cam.ndim == 0:
            cam = np.array([cam.item()])
            logging.info("Cam was a scalar, converted to array.")

        return cam
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()

class HanoiGradCAMAnalyzer:
    """Specialised Grad-CAM analyzer for Towers of Hanoi."""
    def __init__(self, model_path, device, n_action, lr, TD_return):
        self.device = device
        self.env = TowersOfHanoi(N=3, max_steps=200)
        self.model = self._load_model(model_path, n_action, lr, TD_return, device)

    def _load_model(self, model_path, n_action, lr, TD_return, dev):
        """Load the trained MuZero model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        
        networks = MuZeroNet(
            rpr_input_s=self.env.oneH_s_size, action_s=n_action, lr=lr, TD_return=TD_return, device=dev
        ).to(dev)
        model_dict = torch.load(
            f"results/Hanoi/1/muzero_model.pt"
        )

        saved_state_dict = model_dict["Muzero_net"]
    
        # Filter out any compilation-related keys
        clean_state_dict = {}
        for key, value in saved_state_dict.items():
            if not key.startswith('_'):  # Skip internal PyTorch keys
                clean_state_dict[key] = value
        
        try:
            networks.load_state_dict(clean_state_dict)
        except RuntimeError as e:
            print(f"Failed to load state dict: {e}")
            print("Available keys in saved model:", list(saved_state_dict.keys()))
            print("Expected keys in new model:", list(networks.state_dict().keys()))
            raise

        # networks.load_state_dict(model_dict["Muzero_net"])
        # networks.optimiser.load_state_dict(model_dict["Net_optim"])

        networks.to(self.device)
        networks.eval()
        return networks
    
    def analyse_state(self, state, target_layer='representation_net.0', save_dir='gradcam_results'):
        """Analyze a given state using Grad-CAM."""
        os.makedirs(save_dir, exist_ok=True)

        if isinstance(state, tuple):
            state_idx = self.env.states.index(state)

            if state_idx >= self.env.oneH_s_size:
                logging.error(f"State index {state_idx} >= one-hot size {self.env.oneH_s_size}")
                return {}

            # Create one-hot encoding
            one_hot_state = np.zeros(self.env.oneH_s_size)
            one_hot_state[state_idx] = 1.0
            state_tensor = torch.tensor(one_hot_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        results = {}

        try:
            policy_gradcam = GradCAM(self.model, target_layer)
            policy_cam, predicted_action = policy_gradcam.generate_cam(state_tensor, network_type='policy')
            results['policy'] = {
                'cam': policy_cam,
                'predicted_action': predicted_action,
                'description': f'Policy prediction (action {predicted_action})'
            }
        except Exception as e:
            logging.error(f"Policy Grad-CAM failed: {e}")
        finally:
            policy_gradcam.cleanup()

        # Analyse value network
        try:
            value_gradcam = GradCAM(self.model, target_layer)
            value_cam, _ = value_gradcam.generate_cam(state_tensor, network_type='value')
            results['value'] = {
                'cam': value_cam,
                'description': 'Value prediction'
            }
        except Exception as e:
            logging.error(f"Value Grad-CAM failed: {e}")
        finally:
            value_gradcam.cleanup()

        self._visualise_results(state, state_tensor, results, save_dir)
        return results
    
    def _visualise_results(self, original_state, state_tensor, results, save_dir):
        """Create and save visualisations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original state visualization
        # axes[0, 0].imshow(state_tensor.cpu().numpy().reshape(3, 3), cmap='viridis', aspect='auto')
        # axes[0, 0].set_title(f'Original State: {original_state}')
        # axes[0, 0].set_xlabel('Position (Rod A, Rod B, Rod C for each disk)')
        # axes[0, 0].set_ylabel('Disk')
        self._visualize_hanoi_state(axes[0, 0], original_state, "Original State")
        
        # Policy Grad-CAM
        if 'policy' in results:
            policy_cam = results['policy']['cam']
            if len(policy_cam.shape) == 1:
                policy_cam = policy_cam.reshape(3, 3)
            im1 = axes[0, 1].imshow(policy_cam, cmap='jet', alpha=0.7, aspect='auto')
            axes[0, 1].set_title(f"Policy Grad-CAM\n{results['policy']['description']}")
            axes[0, 1].set_xlabel('Position')
            axes[0, 1].set_ylabel('Disk')
            plt.colorbar(im1, ax=axes[0, 1])
        
        # Value Grad-CAM
        if 'value' in results:
            value_cam = results['value']['cam']
            if len(value_cam.shape) == 1:
                value_cam = value_cam.reshape(3, 3)
            im2 = axes[1, 0].imshow(value_cam, cmap='jet', alpha=0.7, aspect='auto')
            axes[1, 0].set_title(f"Value Grad-CAM\n{results['value']['description']}")
            axes[1, 0].set_xlabel('Position')
            axes[1, 0].set_ylabel('Disk')
            plt.colorbar(im2, ax=axes[1, 0])
        
        # Combined overlay
        if 'policy' in results and 'value' in results:
            # Overlay policy and value CAMs
            policy_cam = results['policy']['cam']
            value_cam = results['value']['cam']
            if len(policy_cam.shape) == 1:
                policy_cam = policy_cam.reshape(3, 3)
            if len(value_cam.shape) == 1:
                value_cam = value_cam.reshape(3, 3)
            
            # Normalize and combine
            combined = 0.5 * policy_cam + 0.5 * value_cam
            im3 = axes[1, 1].imshow(combined, cmap='jet', alpha=0.7, aspect='auto')
            axes[1, 1].set_title('Combined Policy + Value Grad-CAM')
            axes[1, 1].set_xlabel('Position')
            axes[1, 1].set_ylabel('Disk')
            plt.colorbar(im3, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # Save the plot
        state_str = str(original_state).replace('(', '').replace(')', '').replace(', ', '_')
        save_path = os.path.join(save_dir, f'gradcam_analysis_{state_str}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Grad-CAM analysis saved to {save_path}")

    def _visualize_hanoi_state(self, ax, state, title):
        """Visualize the Hanoi state as actual towers with disks"""
        # Create a visual representation of the towers
        # State format: (small_disk_rod, medium_disk_rod, large_disk_rod)
        # Rod numbers: 0=A, 1=B, 2=C
        
        rod_positions = [0, 1, 2]  # X positions for rods A, B, C
        disk_sizes = [0.8, 0.6, 0.4]  # Large, Medium, Small disk widths
        disk_colors = ['red', 'blue', 'green']  # Large, Medium, Small disk colors
        disk_names = ['Large', 'Medium', 'Small']
        
        # Clear the axis
        ax.clear()
        
        # Draw the base and rods
        ax.hlines(0, -0.5, 2.5, colors='black', linewidth=3)  # Base
        for rod_pos in rod_positions:
            ax.vlines(rod_pos, 0, 3.5, colors='black', linewidth=2)  # Rods
        
        # Count disks on each rod to determine stacking height
        rod_counts = [0, 0, 0]
        
        # Draw disks (largest to smallest for proper stacking)
        for disk_idx in [0, 1, 2]:  # Large, Medium, Small
            rod = state[2-disk_idx]  # state is (small, medium, large), so reverse index
            
            # Calculate vertical position (stack height)
            y_pos = 0.3 + rod_counts[rod] * 0.4
            rod_counts[rod] += 1
            
            # Draw the disk as a rectangle
            disk_width = disk_sizes[disk_idx]
            rect = plt.Rectangle((rod - disk_width/2, y_pos), disk_width, 0.3, 
                            facecolor=disk_colors[disk_idx], 
                            edgecolor='black', linewidth=1,
                            label=f'{disk_names[disk_idx]} Disk')
            ax.add_patch(rect)
            
            # Add disk label
            ax.text(rod, y_pos + 0.15, disk_names[disk_idx][0], 
                ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Customize the plot
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(-0.1, 3.5)
        ax.set_xticks(rod_positions)
        ax.set_xticklabels(['Rod A', 'Rod B', 'Rod C'])
        ax.set_ylabel('Height')
        ax.set_title(f'{title}: {state}')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=8)

    def analyse_multiple_states(self, states, save_dir='gradcam_results'):
        """Analyse multiple states"""
        all_results = {}

        for i, state in enumerate(states):
            logging.info(f"Analyzing state {i+1}/{len(states)}: {state}")
            results = self.analyse_state(state, save_dir=save_dir)
            all_results[state] = results
        
        self._create_comparison_plot(states, all_results, save_dir)

        return all_results
    
    def _create_comparison_plot(self, states, all_results, save_dir):
        """Create a comparison plot of multiple states"""
        n_states = len(states)
        fig, axes = plt.subplots(n_states, 3, figsize=(15, 5 * n_states))

        if n_states == 1:
            axes = axes.reshape(1, -1)

        for i, state in enumerate(states):
            results = all_results[state]

            if isinstance(state, tuple):
                state_idx = self.env.states.index(state)
                one_hot_state = np.zeros(self.env.oneH_s_size)
                one_hot_state[state_idx] = 1.0
                state_tensor = torch.tensor(one_hot_state, dtype=torch.float32)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32)
            # axes[i, 0].imshow(state_tensor.numpy().reshape(3, 3), cmap='viridis', alpha=0.7, aspect='auto')
            # axes[i, 0].set_title(f'Original State: {state}')
            # axes[i, 0].set_xlabel('Position (Rod A, Rod B, Rod C for each disk)')
            # axes[i, 0].set_ylabel('Disk')
            self._visualize_hanoi_state(axes[i, 0], state, f"State {i+1}")

            if 'policy' in results:
                policy_cam = results['policy']['cam']
                if len(policy_cam.shape) == 1:
                    policy_cam = policy_cam.reshape(3, 3)
                axes[i, 1].imshow(policy_cam, cmap='jet', alpha=0.7, aspect='auto')
                axes[i, 1].set_title(f"Policy Grad-CAM\n{results['policy']['predicted_action']}")

            if 'value' in results:
                value_cam = results['value']['cam']
                if len(value_cam.shape) == 1:
                    value_cam = value_cam.reshape(3, 3)
                axes[i, 2].imshow(value_cam, cmap='jet', alpha=0.7, aspect='auto')
                axes[i, 2].set_title("Value Grad-CAM")
        
        plt.tight_layout()
        comparison_path = os.path.join(save_dir, 'gradcam_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Comparison plot saved to {comparison_path}")

def main():
    parser = argparse.ArgumentParser(description='Run Grad-CAM analysis on MuZero Hanoi model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--save_dir', type=str, default='gradcam_results', help='Directory to save results')
    parser.add_argument('--layer', type=str, default='representation_net.0', help='Target layer for Grad-CAM')
    parser.add_argument('--custom_states', type=str, nargs='*', help='Custom states to analyze (format: "0,0,0")')
    parser.add_argument("--lr" , type=float, default=0, help="Learning rate (default: 0)")
    parser.add_argument(
        "--TD_return",
        type=bool,
        default=True,
        help="Use TD return (default: True)",
    )
    parser.add_argument("--n_action", type=int, default=6, help="Number of actions (default: 6)")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize analyzer
    analyser = HanoiGradCAMAnalyzer(args.model_path, device, args.n_action, args.lr, args.TD_return)

    logging.info(f"Environment has {analyser.env.oneH_s_size} states")
    logging.info(f"First few states: {analyser.env.states[:10]}")
    
    # Define states to analyze
    if args.custom_states:
        states = []
        for state_str in args.custom_states:
            state = tuple(map(int, state_str.split(',')))
            if state in analyser.env.states:
                states.append(state)
            else:
                logging.warning(f"Skipping invalid state: {state}")
    else:
        # Default interesting states
        states = [
            (0, 0, 0),  # Initial state
            (2, 2, 2),  # Goal state
            (1, 0, 0),  # Intermediate state 1
            (0, 1, 0),  # Intermediate state 2
            (2, 1, 0),  # Intermediate state 3
        ]

    states = [(0, 0, 0)]
    
    # Run analysis
    logging.info(f"Analyzing {len(states)} states using layer: {args.layer}")
    results = analyser.analyse_multiple_states(states, save_dir=args.save_dir)
    
    logging.info(f"Analysis complete. Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()
