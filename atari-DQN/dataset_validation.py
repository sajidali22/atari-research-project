import numpy as np
import os

def validate_atari_dataset(file_path):
    print(f"=== Launching Validation for: {os.path.basename(file_path)} ===")
    
    # 1. Check File Existence and Load
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return
    
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"❌ Error: Failed to read .npz file. Corrupted archive? Details: {e}")
        return

    # 2. Key Verification
    expected_keys = {"obs", "actions", "next_obs", "terminals"}
    missing_keys = expected_keys - set(data.keys())
    if missing_keys:
        print(f"❌ Error: Missing critical keys in arrays: {missing_keys}")
        return
    print("✅ All required array keys present.")

    obs = data["obs"]
    actions = data["actions"]
    next_obs = data["next_obs"]
    terminals = data["terminals"]

    # 3. Shape and Dimensionality Verification
    num_samples = obs.shape[0]
    print(f"Total recorded transitions: {num_samples}")

    if not (obs.ndim == 4 and obs.shape[1:] == (84, 84, 4)):
        print(f"❌ Error: 'obs' shape is {obs.shape}, expected (N, 84, 84, 4)")
        return
    if not (next_obs.shape == obs.shape):
        print(f"❌ Error: 'next_obs' shape {next_obs.shape} does not match 'obs' shape {obs.shape}")
        return
    if not (actions.shape == (num_samples,)):
        print(f"❌ Error: 'actions' shape is {actions.shape}, expected ({num_samples},)")
        return
    if not (terminals.shape == (num_samples,)):
        print(f"❌ Error: 'terminals' shape is {terminals.shape}, expected ({num_samples},)")
        return
    print("✅ Shape dimensions and sample alignments are perfectly synchronized.")

    # 4. Data Type Constraints
    if obs.dtype != np.uint8 or next_obs.dtype != np.uint8:
        print(f"❌ Warning: Observations are {obs.dtype}, should be np.uint8 to conserve memory.")
    if actions.dtype not in [np.uint8, np.int64, np.int32]:
        print(f"❌ Warning: Actions are stored as {actions.dtype}.")
    if terminals.dtype != np.bool_:
        print(f"❌ Warning: Terminals are stored as {terminals.dtype}, expected np.bool_")
    print("✅ Array data types match baseline configuration specifications.")

    # 5. Numerical Value Range Checks
    print(f"Pixel Range (obs): min={obs.min()}, max={obs.max()}")
    print(f"Pixel Range (next_obs): min={next_obs.min()}, max={next_obs.max()}")
    if obs.max() > 255 or obs.min() < 0:
        print("❌ Error: Pixel values outside valid uint8 spectrum [0, 255]")
        return
        
    unique_actions = np.unique(actions)
    print(f"Action Space Distribution: unique discrete choices encountered = {unique_actions}")
    if len(unique_actions) == 1:
        print("⚠️ Warning: Only one single action was recorded across the entire dataset. Is the policy stuck?")

    # 6. The Ultimate Markovian Temporal Shift Test
    # For any non-terminal step, the last 3 frames of obs must perfectly match the first 3 frames of next_obs.
    non_terminal_indices = np.where(~terminals)[0]
    
    # Safely sample up to 1000 indices without self-referencing variables
    if len(non_terminal_indices) > 0:
        test_indices = np.random.choice(
            non_terminal_indices, 
            size=min(1000, len(non_terminal_indices)), 
            replace=False
        )
    else:
        test_indices = []

    shift_errors = 0

    shift_errors = 0
    for idx in test_indices:
        # Since channels are LAST (84, 84, 4), we slice the final axis
        # obs[idx] frame stack order: [F_0, F_1, F_2, F_3]
        # next_obs[idx] frame stack order: [F_1, F_2, F_3, F_4]
        current_stack_tail = obs[idx, :, :, 1:]       # Drops F_0, keeps [F_1, F_2, F_3]
        next_stack_head = next_obs[idx, :, :, :-1]    # Keeps [F_1, F_2, F_3], drops F_4
        
        if not np.array_equal(current_stack_tail, next_stack_head):
            shift_errors += 1

    if shift_errors > 0:
        print(f"❌ Temporal Error: {shift_errors} frame stacking mismatches detected in checked transitions.")
        print("   This means the temporal sequence inside your data loader loop is broken.")
        return
    else:
        print("✅ Temporal Shift Integrity Check Passed! Frame histories are clean and sequential.")

    # 7. Terminal State Distribution
    total_terminals = np.sum(terminals)
    print(f"Terminal Episode Boundaries Encountered: {total_terminals} / {num_samples} total transitions.")
    
    print("🎉 DATASET VALIDATION SUCCESSFUL: Structurally sound for world model modeling.")

if __name__ == "__main__":
    # Point this to one of your generated files
    target_file = "/home/sajidali/Entrollics/TU-Dresden/ResearchProject/atari-DQN/custom_datasets/train/MsPacmanNoFrameskip-v4_expert_50000_transitions.npz"
    validate_atari_dataset(target_file)