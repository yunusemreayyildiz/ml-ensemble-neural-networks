# Task 1.2 Multi-Layer Perceptron 
import warnings # to ignore convergence warnings(icreasing readability of output)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA 

# define a function to add noise to the data
# Added random_seed parameter for reproducibility
def add_noise(data, noise_percentage, random_seed=None):
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random

    n_samples, n_features = data.shape
    data_noisy = data.copy()
    n_noisy_samples = int(n_samples * (noise_percentage / 100)) #calculates number of samples to be noised
    
    # Randomly select indices of samples to corrupt
    noisy_indices = rng.choice(n_samples, n_noisy_samples, replace=False)
    
    for index in noisy_indices:
        # Randomly choose 10 features (pixels) to modify for this sample
        feature_indices = rng.choice(n_features, 10, replace=False)
        # Apply the noise formula given in the assignment |value - 16|
        # This effectively inverts the pixel color in a 0-16 scale
        for f_index in feature_indices:
            data_noisy[index, f_index] = abs(data_noisy[index, f_index] - 16)
    return data_noisy

def run_scenarios(X, y, scenarios_list):
    print("Scenario Train Noise  Test Noise Train Error  Test Error")
    GLOBAL_SEED = 42

    for code, train_noise_pct, test_noise_pct in scenarios_list:
        # Split the data (70% Train, 30% Test)for each scenario
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=GLOBAL_SEED)
        
        # Apply noise to training set if required
        if train_noise_pct > 0:
            X_train = add_noise(X_train, train_noise_pct, random_seed=GLOBAL_SEED)
        # Apply noise to test set if required
        if test_noise_pct > 0:
            # Using seed+1 for test set to ensure different noise pattern but stability
            X_test = add_noise(X_test, test_noise_pct, random_seed=GLOBAL_SEED + 1)
            
        # Initialize and train the MLP Classifier
        # Using default hidden_layer_sizes since not specified for this task
        # Using default layer_size =100
        mlp = MLPClassifier(max_iter=500, random_state=GLOBAL_SEED)
        mlp.fit(X_train, y_train)
        
        # Calculate Error (Error = 1.0 - Accuracy)for a test and 
        test_error = 1.0 - mlp.score(X_test, y_test)
        
        print(f"{code}    {train_noise_pct}     {test_noise_pct}    {test_error:<15.4f}")

def run_denoising_experiment(X, y):
    print("\nScenario h: 50% Noise")
    GLOBAL_SEED = 42
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=GLOBAL_SEED)
    # Apply 50% noise to both sets
    X_train_noisy = add_noise(X_train, 50, random_seed=GLOBAL_SEED)
    X_test_noisy = add_noise(X_test, 50, random_seed=GLOBAL_SEED + 1)
    
    # Apply PCA for denoising
    # Keeping 80% of variance to filter out noise (high frequency components) - Adjusted for better results
    pca = PCA(n_components=0.80, random_state=GLOBAL_SEED)# PCA instance to retain variance
    pca.fit(X_train_noisy)#Prıncipal component analysis 
    
    # Transform (denoise) the data
    X_train_denoised = pca.transform(X_train_noisy)# denoised training data
    X_test_denoised = pca.transform(X_test_noisy)# denoised testing data
    
    # Train MLP on denoised data
    # Note: Input dimension is reduced after PCA, MLP handles this automatically
    mlp_denoised = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=GLOBAL_SEED)
    mlp_denoised.fit(X_train_denoised, y_train)
    
    # Calculate errors
    denoised_test_error = 1.0 - mlp_denoised.score(X_test_denoised, y_test)
    
    print(f"{'j (Denoised)'} {'50'} {'50'}  {denoised_test_error:<15.4f}")

def visualize_noise_effect(X):  
    # Randomly select a tuple from D (using random state for reproducibility)
    rng = np.random.RandomState(42)
    random_index = rng.randint(0, len(X))
    sample = X[random_index] # Take a random sample
    
    # Manually add noise to 10 pixels for visualization
    # We use our function to keep logic consistent
    sample_reshaped = sample.reshape(1, -1)
    # 100% noise rate ensures this specific sample gets processed
    sample_noisy = add_noise(sample_reshaped, 100, random_seed=42).flatten()
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))# visualize original and noisy form comparative by using 1*2 
    
    axes[0].imshow(sample.reshape(8, 8), cmap='gray_r')#put original image
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(sample_noisy.reshape(8, 8), cmap='gray_r')# put noisy image
    axes[1].set_title("Noisy")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('Noising.pdf')
    plt.show(block = False)
    plt.close() 
if __name__ == "__main__":
    # load digit dataset
    digits = load_digits()
    X_data = digits.data
    y_data = digits.target
    noise_scenarios = [
        ('a', 0, 25),
        ('b', 0, 50),
        ('c', 0, 75),
        ('d', 25, 0),
        ('e', 50, 0),
        ('f', 75, 0),
        ('g', 25, 25),
        ('h', 50, 50),
        ('i', 75, 75)
    ]
    run_scenarios(X_data, y_data, noise_scenarios)
    run_denoising_experiment(X_data, y_data)
    visualize_noise_effect(X_data)